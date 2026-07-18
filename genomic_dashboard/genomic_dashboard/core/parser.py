from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import Iterator, List, Optional

from Bio.SeqIO.FastaIO import SimpleFastaParser
from Bio.SeqIO.QualityIO import FastqGeneralIterator

# Setup lightweight logging for dropped records
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
        # FASTA files are generally simpler structurally, but we insulate it similarly
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
                # Manually advancing the iterator allows us to trap errors 
                # inside the loop structure before it kills the stream context.
                res = next(fastq_parser, None)
                if res is None:
                    break
                
                title, seq, qual = res
                
                # Internal sanity check: FASTQ sequence and quality lengths must match
                if len(seq) != len(qual):
                    raise ValueError(f"Sequence length ({len(seq)}) and Quality length ({len(qual)}) mismatch.")
                
                record_id = title.split(None, 1)[0]
                yield SequenceRecord(id=record_id, sequence=seq, quality=qual)
                
            except (ValueError, IndexError, AssertionError) as e:
                # Malformed Record Recovery KPI: Log defect, skip block, keep streaming
                logger.error(f"Skipping malformed FASTQ record. Reason: {e}")
                continue
            except Exception as e:
                # Catch-all for unexpected low-level errors to guarantee stream survival
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
