"""
High-throughput, memory-safe FASTA/FASTQ parsing.

Design goals
------------
* Never write the uploaded file to disk — everything happens in an
  in-memory buffer (io.StringIO / io.BytesIO) so multi-GB uploads don't
  fill up ephemeral container storage.
* Stream records one at a time (generator) instead of loading the whole
  file into a list, so peak memory stays roughly constant regardless of
  file size.
* Use Biopython's low-level SimpleFastaParser / FastqGeneralIterator,
  which are much faster and lighter than SeqIO.parse() for large files
  because they skip building full SeqRecord objects until needed.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Iterator, List, Optional

from Bio.SeqIO.FastaIO import SimpleFastaParser
from Bio.SeqIO.QualityIO import FastqGeneralIterator

# Setup lightweight logging for dropped or mismatched records
logger = logging.getLogger(__name__)


@dataclass
class SequenceRecord:
    id: str
    sequence: str
    quality: Optional[str] = None

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def gc_content(self) -> float:
        if not self.sequence:
            return 0.0
        seq = self.sequence.upper()
        gc = seq.count("G") + seq.count("C")
        return round(100 * gc / len(seq), 2)
    
    @property
    def error_probabilities(self) -> List[float]:
        """
        Decodes the FASTQ quality string into numeric error probabilities.
        Fidelity of Quality Score Mapping KPI.
        """
        if not self.quality:
            return []
        # Standard Phred+33 encoding conversion to p-value
        return [10.0 ** (- (ord(char) - 33) / 10.0) for char in self.quality]


@dataclass
class PairedEndStreamStats:
    total_pairs: int = 0
    mismatched_pairs: int = 0

    @property
    def integrity_rate(self) -> float:
        """
        Calculates the Paired-End Integrity Rate KPI.
        Returns a percentage value between 0.0 and 100.0.
        """
        if self.total_pairs == 0:
            return 100.0
        matched = self.total_pairs - self.mismatched_pairs
        return round((matched / self.total_pairs) * 100, 2)


def _sniff_format(text_head: str) -> str:
    stripped = text_head.lstrip()
    if stripped.startswith(">"):
        return "fasta"
    if stripped.startswith("@"):
        return "fastq"
    raise ValueError(
        "Unrecognized sequence format — file must start with '>' (FASTA) "
        "or '@' (FASTQ)."
    )


def iter_records_from_bytes(raw_bytes: bytes) -> Iterator[SequenceRecord]:
    """
    Stream SequenceRecord objects from raw uploaded bytes without ever
    touching disk. Robust against isolated malformed records.
    """
    head = raw_bytes[:1024].decode("utf-8", errors="ignore")
    fmt = _sniff_format(head)

    text_buffer = io.StringIO(raw_bytes.decode("utf-8", errors="ignore"))

    if fmt == "fasta":
        fasta_parser = SimpleFastaParser(text_buffer)
        while True:
            try:
                res = next(fasta_parser, None)
                if res is None:
                    break
                title, seq = res
                record_id = title.split(None, 1)[0]
                yield SequenceRecord(id=record_id, sequence=seq)
            except Exception as e:
                logger.error(f"Skipping malformed FASTA record: {e}")
                continue
    else:
        fastq_parser = FastqGeneralIterator(text_buffer)
        while True:
            try:
                # Advancing the iterator manually to trap parsing errors 
                # inside the loop block before it terminates the generator context.
                res = next(fastq_parser, None)
                if res is None:
                    break
                
                title, seq, qual = res
                
                # Structural check: FASTQ sequence and quality lengths must match
                if len(seq) != len(qual):
                    raise ValueError(f"Sequence length ({len(seq)}) and Quality length ({len(qual)}) mismatch.")
                
                record_id = title.split(None, 1)[0]
                yield SequenceRecord(id=record_id, sequence=seq, quality=qual)
                
            except (ValueError, IndexError, AssertionError) as e:
                # Malformed Record Recovery KPI: Log defect, skip block, keep streaming
                logger.error(f"Skipping malformed FASTQ record. Reason: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error parsing record: {e}")
                continue


def stream_with_progress(raw_bytes: bytes, progress_callback=None, report_every: int = 500):
    """
    Wraps iter_records_from_bytes and invokes progress_callback(n_records,
    approx_bytes_consumed) periodically, so a Streamlit progress bar /
    status text can update during long parses without materializing the
    full record list first.
    """
    total_size = max(len(raw_bytes), 1)
    consumed = 0
    count = 0

    for record in iter_records_from_bytes(raw_bytes):
        count += 1
        # Approximate bytes consumed by this record (id + seq + quality + headers)
        consumed += len(record.id) + record.length + (len(record.quality) if record.quality else 0) + 4

        if progress_callback and count % report_every == 0:
            pct = min(consumed / total_size, 1.0)
            progress_callback(count, pct)

        yield record

    if progress_callback:
        progress_callback(count, 1.0)


def iter_paired_end(
    raw_bytes_r1: bytes, 
    raw_bytes_r2: bytes, 
    stats_callback=None
) -> Iterator[tuple[SequenceRecord, SequenceRecord]]:
    """
    Simultaneously streams Forward (R1) and Reverse (R2) records in memory.
    Flags ID mismatches on the fly to track the Paired-End Integrity Rate.
    
    Optional stats_callback: a callable that accepts a PairedEndStreamStats object
    to update dashboards/logs at the end of the run.
    """
    stats = PairedEndStreamStats()
    
    stream_r1 = iter_records_from_bytes(raw_bytes_r1)
    stream_r2 = iter_records_from_bytes(raw_bytes_r2)

    # zip terminates safely as soon as the shorter stream runs out
    for rec1, rec2 in zip(stream_r1, stream_r2):
        stats.total_pairs += 1
        
        # Strip common Illumina/Sanger paired-end suffixes to isolate core ID
        # e.g., "cluster_1/1" vs "cluster_1/2" or "id_1 1:N:0:1" vs "id_1 2:N:0:1"
        id1_clean = rec1.id.split('/')[0].split()[0]
        id2_clean = rec2.id.split('/')[0].split()[0]
        
        if id1_clean != id2_clean:
            stats.mismatched_pairs += 1
            logger.warning(
                f"Paired-End Integrity Violation at pair index {stats.total_pairs}: "
                f"R1 ID '{rec1.id}' does not match R2 ID '{rec2.id}'."
            )
            
        yield rec1, rec2

    if stats_callback:
        stats_callback(stats)
