"""
3D protein structure viewer (e.g. for AlphaFold / ESMFold PDB output),
embedded via the NGL.js viewer library over CDN.
"""

import json

import streamlit.components.v1 as components


def render_structure(pdb_text: str, height: int = 500):
    """
    pdb_text: raw contents of a .pdb file (as produced by AlphaFold/ESMFold).
    """
    safe_pdb = json.dumps(pdb_text)

    html = f"""
    <div id="viewport" style="width:100%; height:{height}px;"></div>
    <script src="https://cdn.jsdelivr.net/gh/nglviewer/ngl@v2.0.0-dev.39/dist/ngl.js"></script>
    <script>
      const pdbData = {safe_pdb};
      const stage = new NGL.Stage("viewport", {{ backgroundColor: "white" }});

      window.addEventListener("resize", () => stage.handleResize(), false);

      const blob = new Blob([pdbData], {{ type: "text/plain" }});
      stage.loadFile(blob, {{ ext: "pdb" }}).then(function (component) {{
        component.addRepresentation("cartoon", {{ colorScheme: "residueindex" }});
        component.autoView();
      }});
    </script>
    """
    components.html(html, height=height + 20, scrolling=False)


def render_structure_from_url(pdb_url: str, height: int = 500):
    """
    Convenience variant when you already have a hosted PDB URL
    (e.g. an AlphaFold DB entry) rather than raw text.
    """
    safe_url = json.dumps(pdb_url)

    html = f"""
    <div id="viewport" style="width:100%; height:{height}px;"></div>
    <script src="https://cdn.jsdelivr.net/gh/nglviewer/ngl@v2.0.0-dev.39/dist/ngl.js"></script>
    <script>
      const stage = new NGL.Stage("viewport", {{ backgroundColor: "white" }});
      window.addEventListener("resize", () => stage.handleResize(), false);
      stage.loadFile({safe_url}).then(function (component) {{
        component.addRepresentation("cartoon", {{ colorScheme: "residueindex" }});
        component.autoView();
      }});
    </script>
    """
    components.html(html, height=height + 20, scrolling=False)
