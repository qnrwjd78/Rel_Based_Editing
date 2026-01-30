import argparse
import base64
import json
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, List, Optional, Tuple

try:
    import cv2
except Exception:
    cv2 = None


def _load_prev_condition(path: str) -> Dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_prev_value(prev: Dict, key: str):
    return prev.get(key) if prev and key in prev else None


def _prompt_text(label: str) -> str:
    return input(f"{label} 입력: ").strip()


def _select_bbox_cv2(image_path: str, label: str) -> List[int]:
    if cv2 is None:
        raise RuntimeError("cv2 not available")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지 로드 실패: {image_path}")
    r = cv2.selectROI(label, img, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(label)
    x, y, w, h = r
    if w <= 0 or h <= 0:
        raise ValueError(f"{label} bbox 선택이 비어 있습니다.")
    return [int(x), int(y), int(x + w), int(y + h)]


def _select_bbox_manual(label: str) -> List[int]:
    raw = input(f"{label} bbox 입력 (x1,y1,x2,y2): ").strip()
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox 입력 형식이 올바르지 않습니다.")
    vals = [int(float(p)) for p in parts]
    return vals


def _select_bbox(image_path: str, label: str, ui_mode: str) -> List[int]:
    if ui_mode == "manual":
        return _select_bbox_manual(label)
    if cv2 is not None and os.environ.get("DISPLAY"):
        try:
            return _select_bbox_cv2(image_path, label)
        except Exception:
            if ui_mode == "auto":
                return _select_bbox_manual(label)
            raise
    if ui_mode == "auto":
        return _select_bbox_manual(label)
    raise RuntimeError("UI 모드가 web인데 단일 bbox 선택을 호출했습니다.")


def _pick_port(start: int = 7860, end: int = 7890) -> int:
    for port in range(start, end + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("사용 가능한 포트를 찾지 못했습니다.")


def _build_html(image_b64: str, orig_w: int, orig_h: int) -> str:
    display_w = min(1024, orig_w)
    scale = display_w / orig_w
    display_h = int(orig_h * scale)
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>bbox selector</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; }}
    #container {{ position: relative; width: {display_w}px; }}
    #img {{ width: {display_w}px; height: {display_h}px; display: block; }}
    #canvas {{ position: absolute; left: 0; top: 0; }}
    .row {{ margin-top: 12px; }}
    button {{ margin-right: 8px; }}
    code {{ background: #f4f4f4; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h3>Draw bbox_s then bbox_o</h3>
  <div id="container">
    <img id="img" src="data:image/png;base64,{image_b64}" />
    <canvas id="canvas" width="{display_w}" height="{display_h}"></canvas>
  </div>
  <div class="row">
    <button onclick="resetAll()">Reset</button>
    <button onclick="submit()">Submit</button>
  </div>
  <div class="row">
    bbox_s: <code id="bbox_s">-</code><br/>
    bbox_o: <code id="bbox_o">-</code>
  </div>
  <script>
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const scale = {scale};
    let drawing = false;
    let startX = 0, startY = 0;
    let rects = {{ s: null, o: null }};

    function drawAll(tempRect=null) {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 2;
      if (rects.s) {{
        ctx.strokeStyle = 'blue';
        ctx.strokeRect(rects.s.x, rects.s.y, rects.s.w, rects.s.h);
      }}
      if (rects.o) {{
        ctx.strokeStyle = 'red';
        ctx.strokeRect(rects.o.x, rects.o.y, rects.o.w, rects.o.h);
      }}
      if (tempRect) {{
        ctx.strokeStyle = rects.s ? 'red' : 'blue';
        ctx.strokeRect(tempRect.x, tempRect.y, tempRect.w, tempRect.h);
      }}
    }}

    function updateLabels() {{
      document.getElementById('bbox_s').textContent = rects.s ? JSON.stringify(toOrig(rects.s)) : '-';
      document.getElementById('bbox_o').textContent = rects.o ? JSON.stringify(toOrig(rects.o)) : '-';
    }}

    function toOrig(r) {{
      const x1 = Math.round(r.x / scale);
      const y1 = Math.round(r.y / scale);
      const x2 = Math.round((r.x + r.w) / scale);
      const y2 = Math.round((r.y + r.h) / scale);
      return [x1,y1,x2,y2];
    }}

    canvas.addEventListener('mousedown', (e) => {{
      if (rects.s && rects.o) return;
      drawing = true;
      const rect = canvas.getBoundingClientRect();
      startX = e.clientX - rect.left;
      startY = e.clientY - rect.top;
    }});

    canvas.addEventListener('mousemove', (e) => {{
      if (!drawing) return;
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const w = x - startX;
      const h = y - startY;
      drawAll({{x: startX, y: startY, w: w, h: h}});
    }});

    canvas.addEventListener('mouseup', (e) => {{
      if (!drawing) return;
      drawing = false;
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const w = x - startX;
      const h = y - startY;
      const fixed = {{
        x: Math.min(startX, x),
        y: Math.min(startY, y),
        w: Math.abs(w),
        h: Math.abs(h)
      }};
      if (fixed.w < 2 || fixed.h < 2) return;
      if (!rects.s) rects.s = fixed;
      else if (!rects.o) rects.o = fixed;
      drawAll();
      updateLabels();
    }});

    function resetAll() {{
      rects = {{ s: null, o: null }};
      drawAll();
      updateLabels();
    }}

    async function submit() {{
      if (!rects.s || !rects.o) {{
        alert('bbox_s와 bbox_o를 모두 선택하세요.');
        return;
      }}
      const payload = {{
        bbox_s: toOrig(rects.s),
        bbox_o: toOrig(rects.o)
      }};
      const res = await fetch('/submit', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify(payload)
      }});
      if (res.ok) {{
        document.body.innerHTML = '<h3>Saved. You can close this tab.</h3>';
      }} else {{
        alert('Submit failed');
      }}
    }}
  </script>
</body>
</html>"""


def _select_bboxes_web(image_path: str) -> Tuple[List[int], List[int]]:
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    from PIL import Image

    img = Image.open(image_path)
    orig_w, orig_h = img.size

    html = _build_html(image_b64, orig_w, orig_h)
    result: Dict[str, List[int]] = {}
    done = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(html.encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):  # noqa: N802
            if self.path != "/submit":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(payload)
                result["bbox_s"] = data.get("bbox_s")
                result["bbox_o"] = data.get("bbox_o")
            except Exception:
                self.send_response(400)
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            done.set()

        def log_message(self, format, *args):  # noqa: A002
            return

    port = _pick_port()
    server = HTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"브라우저에서 열기: http://127.0.0.1:{port}")
    done.wait()
    server.shutdown()
    thread.join()

    if "bbox_s" not in result or "bbox_o" not in result:
        raise RuntimeError("웹 UI에서 bbox를 받지 못했습니다.")
    return result["bbox_s"], result["bbox_o"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", required=True, help="이미지 경로")
    parser.add_argument("--cond_path", required=False, help="이전 condition json 경로", default=None,)
    parser.add_argument(
        "--out_path",
        default=None,
        help="저장할 condition json 경로 (기본: cond_path 덮어쓰기)",
    )
    parser.add_argument(
        "--ui",
        choices=["auto", "web", "manual"],
        default="auto",
        help="bbox 입력 방식: auto/cv2, web, manual",
    )
    args = parser.parse_args()

    prev = _load_prev_condition(args.cond_path)

    s_word = _prompt_text("s (subject)")
    a_word = _prompt_text("a (action)")
    o_word = _prompt_text("o (object)")

    if args.ui == "web" or (args.ui == "auto" and (cv2 is None or not os.environ.get("DISPLAY"))):
        bbox_s, bbox_o = _select_bboxes_web(args.img_path)
    else:
        print("bbox_s 선택 창이 뜹니다. 드래그로 선택하세요.")
        bbox_s = _select_bbox(args.img_path, "bbox_s", args.ui)
        print("bbox_o 선택 창이 뜹니다. 드래그로 선택하세요.")
        bbox_o = _select_bbox(args.img_path, "bbox_o", args.ui)

    out = {
        "fname": _get_prev_value(prev, "fname") or "",
        "inv_prompt": _get_prev_value(prev, "inv_prompt") or "",
        "bbox_s_pre": _get_prev_value(prev, "bbox_s_pre") or [],
        "bbox_o_pre": _get_prev_value(prev, "bbox_o_pre") or [],
        "mask_o": _get_prev_value(prev, "mask_o") or "",
        "mask_s": _get_prev_value(prev, "mask_s") or "",
        "s_word": s_word,
        "a_word": a_word,
        "o_word": o_word,
        "bbox_s": bbox_s,
        "bbox_o": bbox_o,
    }

    out_path = args.out_path or args.cond_path
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=4)
    print(f"저장 완료: {out_path}")


if __name__ == "__main__":
    main()
