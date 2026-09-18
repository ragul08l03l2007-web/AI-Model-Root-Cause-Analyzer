# test_server_endpoints.py

import io
import json
import threading
import time
import urllib.request
import urllib.parse
from http.server import ThreadingHTTPServer

import pandas as pd
from app import AppHandler, reset_state


def run_test():
    reset_state()
    server = ThreadingHTTPServer(("localhost", 8081), AppHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.5)

    base_url = "http://localhost:8081"

    try:
        # 1. GET /
        req = urllib.request.urlopen(f"{base_url}/")
        assert req.status == 200, f"Expected 200 on /, got {req.status}"
        html = req.read().decode("utf-8")
        assert "AI Model Root-Cause Analyzer" in html
        print("-> GET / passed (200)")

        # 2. POST /upload
        csv_data = (
            "id,age,income,score,risk\n"
            "1,25,50000,75,0\n"
            "2,35,80000,82,0\n"
            "3,45,60000,45,1\n"
            "4,55,95000,90,0\n"
            "5,22,30000,30,1\n"
            "6,60,110000,88,0\n"
            "7,29,45000,60,0\n"
            "8,41,70000,50,1\n"
            "9,38,72000,55,1\n"
            "10,50,90000,80,0\n"
        ).encode("utf-8")

        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="mode"\r\n\r\n'
            f"auto\r\n"
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="dataset"; filename="test_credit.csv"\r\n'
            f"Content-Type: text/csv\r\n\r\n"
        ).encode("utf-8") + csv_data + f"\r\n--{boundary}--\r\n".encode("utf-8")

        post_req = urllib.request.Request(
            f"{base_url}/upload",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
        )

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        opener = urllib.request.build_opener(NoRedirect)
        try:
            resp = opener.open(post_req)
            status = resp.status
            loc = resp.headers.get("Location")
        except urllib.error.HTTPError as e:
            status = e.code
            loc = e.headers.get("Location")

        assert status == 302 and loc == "/preview", f"Expected 302 -> /preview, got {status} -> {loc}"
        print("-> POST /upload passed (302 -> /preview)")

        # 3. GET /preview
        resp = urllib.request.urlopen(f"{base_url}/preview")
        assert resp.status == 200
        preview_html = resp.read().decode("utf-8")
        assert "test_credit.csv" in preview_html
        assert "income" in preview_html
        print("-> GET /preview passed (200)")

        # 4. POST /analyze
        form_data = urllib.parse.urlencode({"target": "risk", "mode": "auto"}).encode("utf-8")
        analyze_req = urllib.request.Request(
            f"{base_url}/analyze",
            data=form_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        try:
            resp = opener.open(analyze_req)
            status = resp.status
            loc = resp.headers.get("Location")
        except urllib.error.HTTPError as e:
            status = e.code
            loc = e.headers.get("Location")

        assert status == 302 and loc == "/dashboard", f"Expected 302 -> /dashboard, got {status} -> {loc}"
        print("-> POST /analyze passed (302 -> /dashboard)")

        # 5. GET /dashboard
        resp = urllib.request.urlopen(f"{base_url}/dashboard")
        assert resp.status == 200
        dash_html = resp.read().decode("utf-8")
        assert "Multi-Model Comparison" in dash_html
        assert "Diagnostic Signals &amp; Root-Cause Hypotheses" in dash_html or "Diagnostic Signals & Root-Cause Hypotheses" in dash_html or "Evidence-Based Root-Cause Candidates" in dash_html
        assert "np.int" not in dash_html, "NumPy type found in dashboard HTML!"
        print("-> GET /dashboard passed (200, clean HTML with no numpy types)")

        # 6. GET /download/analysis.json
        resp = urllib.request.urlopen(f"{base_url}/download/analysis.json")
        assert resp.status == 200
        json_data = json.loads(resp.read().decode("utf-8"))
        assert "dataset" in json_data and "analysis" in json_data
        print("-> GET /download/analysis.json passed (valid JSON)")

        # 7. GET /download/summary.csv
        resp = urllib.request.urlopen(f"{base_url}/download/summary.csv")
        assert resp.status == 200
        summary_csv = resp.read().decode("utf-8-sig")
        assert "selected_model" in summary_csv
        print("-> GET /download/summary.csv passed")

        # 8. GET /download/prediction-errors.csv
        resp = urllib.request.urlopen(f"{base_url}/download/prediction-errors.csv")
        assert resp.status == 200
        err_csv = resp.read().decode("utf-8-sig")
        assert "row" in err_csv and "actual" in err_csv
        print("-> GET /download/prediction-errors.csv passed")

        # 9. GET /download/remediation_script.py
        resp = urllib.request.urlopen(f"{base_url}/download/remediation_script.py")
        assert resp.status == 200
        script_content = resp.read().decode("utf-8")
        assert len(script_content) > 50
        compile(script_content, "<string>", "exec")
        print("-> GET /download/remediation_script.py passed (valid Python syntax)")

        # 10. POST /api/copilot/chat
        chat_payload = json.dumps({"question": "Why is the model underperforming?"}).encode("utf-8")
        chat_req = urllib.request.Request(
            f"{base_url}/api/copilot/chat",
            data=chat_payload,
            headers={"Content-Type": "application/json"}
        )
        chat_resp = urllib.request.urlopen(chat_req)
        assert chat_resp.status == 200
        chat_json = json.loads(chat_resp.read().decode("utf-8"))
        assert chat_json["status"] == "success" and "reply" in chat_json and len(chat_json["reply"]) > 10
        print(f"-> POST /api/copilot/chat passed ({chat_json['provider']})")

        # 11. GET /back-to-preview
        try:
            resp = opener.open(f"{base_url}/back-to-preview")
            status = resp.status
            loc = resp.headers.get("Location")
        except urllib.error.HTTPError as e:
            status = e.code
            loc = e.headers.get("Location")
        assert status == 302 and loc == "/preview"
        print("-> GET /back-to-preview passed (302 -> /preview)")

        # 10. GET /back-to-upload
        try:
            resp = opener.open(f"{base_url}/back-to-upload")
            status = resp.status
            loc = resp.headers.get("Location")
        except urllib.error.HTTPError as e:
            status = e.code
            loc = e.headers.get("Location")
        assert status == 302 and loc == "/"
        print("-> GET /back-to-upload passed (302 -> /)")

        print("\nALL SERVER ROUTE & ENDPOINT INTEGRATION TESTS PASSED PERFECTLY!")

    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    run_test()
