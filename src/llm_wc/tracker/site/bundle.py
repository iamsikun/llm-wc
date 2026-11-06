"""Pack the built site into one self-contained HTML file for private sharing.

The file holds every page, the styles, the scripts, the data, and the downloads (not the
link-preview images). Opened from
disk, it shows the page named in the URL hash and writes it into the document; links change the
hash and reload. ``fetch`` and download links are answered from the embedded files, so the
file works without a server or a network.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from llm_wc.tracker.paths import BUILD

ASSET = re.compile(r"\?v=[0-9a-f]+")
LINK_CSS = re.compile(r'<link rel="stylesheet" href="[^"]*static/style\.css"[^>]*>')
LINK_ICON = re.compile(r'<link rel="icon"[^>]*>')
SCRIPT_SRC = re.compile(r'<script src="[^"]*static/([a-z]+\.js)"></script>')

SHELL = """<!doctype html>
<meta charset="utf-8">
<script>
(function () {
  "use strict";
  const PAGES = %(pages)s;
  const FILES = %(files)s;
  const STATIC = %(static)s;
  const page = decodeURIComponent(location.hash.slice(1).split("#")[0]) || "index.html";
  const key = PAGES[page] !== undefined ? page : "404.html";
  const dir = key.includes("/") ? key.slice(0, key.lastIndexOf("/") + 1) : "";
  // Resolve a link or request relative to the current page to a site path.
  const resolve = (href) => {
    const u = new URL(href, "https://site.invalid/" + dir);
    return u.origin === "https://site.invalid" ? decodeURIComponent(u.pathname.slice(1)) + u.hash : null;
  };
  const text = (path) => {
    const f = FILES[path];
    return f === undefined ? undefined : f.text;
  };
  window.fetch = (url) => {
    const path = (resolve(String(url)) || "").split("#")[0];
    const body = text(path);
    if (body === undefined) return Promise.resolve(new Response("Not found", { status: 404 }));
    return Promise.resolve(new Response(body, { headers: { "Content-Type": path.endsWith(".json") ? "application/json" : "text/plain" } }));
  };
  history.scrollRestoration = "manual";  // each page opens at the top
  addEventListener("hashchange", () => location.reload());
  document.addEventListener("click", (e) => {
    const a = e.target.closest && e.target.closest("a[href]");
    if (!a) return;
    const href = a.getAttribute("href");
    if (/^(https?:|mailto:)/.test(href)) { a.target = "_blank"; return; }
    if (href.startsWith("#")) return;
    const target = resolve(href);
    if (target === null) return;
    const [path, frag] = target.split("#");
    if (FILES[path] !== undefined) {
      e.preventDefault();
      const f = FILES[path];
      const bytes = new TextEncoder().encode(f.text);
      const link = document.createElement("a");
      link.href = URL.createObjectURL(new Blob([bytes]));
      link.download = path.split("/").pop();
      link.click();
      return;
    }
    if (PAGES[path] !== undefined) {
      e.preventDefault();
      location.hash = path + (frag ? "#" + frag : "");
    }
  }, true);
  let html = PAGES[key]
    .replace(/<link rel="stylesheet" data-inline>/, () => "<style>" + STATIC["style.css"] + "<\\/style>")
    .replace(/<script data-inline="([a-z]+\\.js)"><\\/script>/g, (_, name) => "<script>" + STATIC[name] + "<\\/script>")
    .replace(/(<img[^>]*src=")([^"]+\\.svg)"/g, (m, pre, src) => {
      const p = resolve(src), f = p && FILES[p.split("#")[0]];
      return f ? pre + "data:image/svg+xml;charset=utf-8," + encodeURIComponent(text(p)) + '"' : m;
    });
  document.write(html.replace(/^<!doctype html>/i, ""));
  document.close();
})();
</script>
"""


def _js(value) -> str:
    """JSON that is safe inside a <script> element."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def bundle(site: Path = BUILD, out: Path | None = None) -> Path:
    """Write ``out`` (default ``_bundle/sota-or-luck.html``) from a built ``site`` directory."""
    site = Path(site)
    out = Path(out) if out else site.parent / "_bundle" / "sota-or-luck.html"
    pages = {}
    for p in sorted(site.rglob("*.html")):
        html = ASSET.sub("", p.read_text())
        html = LINK_CSS.sub('<link rel="stylesheet" data-inline>', html)
        html = LINK_ICON.sub("", html)
        html = SCRIPT_SRC.sub(lambda m: f'<script data-inline="{m[1]}"></script>', html)
        pages[p.relative_to(site).as_posix()] = html
    static = {p.name: p.read_text() for p in (site / "static").iterdir() if p.suffix in (".css", ".js")}
    files = {}
    for sub in ("data", "badges"):
        for p in sorted((site / sub).rglob("*")):
            if p.is_file():
                files[p.relative_to(site).as_posix()] = {"text": p.read_text()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(SHELL % {"pages": _js(pages), "files": _js(files), "static": _js(static)})
    return out
