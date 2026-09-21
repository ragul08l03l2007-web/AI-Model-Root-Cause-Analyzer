# test_server_endpoints.py
"""
Integration verification test for app.py REST endpoints and HTML structure using standard library urllib.
"""

import threading
import time
import json
import urllib.request
import urllib.error
from app import ThreadingHTTPServer, AppHandler, HOST, PORT

def http_get(url):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        return response.status, response.headers, response.read()

def http_post_json(url, data):
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as response:
        return response.status, response.headers, response.read()

def run_tests():
    test_port = 8008
    server = ThreadingHTTPServer((HOST, test_port), AppHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"[*] Server started on thread port {test_port}.")
    time.sleep(0.5)

    base_url = f"http://127.0.0.1:{test_port}"

    try:
        # 1. Test GET /
        print("\n--- 1. Testing GET / (HTML) ---")
        status, headers, body = http_get(f"{base_url}/")
        html_text = body.decode("utf-8")
        assert status == 200, f"Expected 200, got {status}"
        assert "Root Cause AI" in html_text, "Title not in HTML"
        assert "plotly-risk-gauge" in html_text, "Plotly gauge container missing"
        assert "config-target-select" in html_text, "Target select missing"
        print("[OK] GET / successfully rendered index.html")

        # 2. Test GET /styles.css and /app.js
        print("\n--- 2. Testing Static Assets ---")
        status, headers, _ = http_get(f"{base_url}/styles.css")
        assert status == 200 and "text/css" in headers.get("Content-Type", "")
        status, headers, _ = http_get(f"{base_url}/app.js")
        assert status == 200 and "javascript" in headers.get("Content-Type", "")
        print("[OK] Static CSS and JS assets served correctly")

        # 3. Test GET /api/sample?type=classification
        print("\n--- 3. Testing GET /api/sample (Classification) ---")
        status, headers, body = http_get(f"{base_url}/api/sample?type=classification")
        assert status == 200, f"Expected 200, got {status}"
        sample_data = json.loads(body.decode("utf-8"))
        assert sample_data.get("filename") == "random_test_dataset.csv"
        assert sample_data.get("rows") == 250
        assert "churn" in sample_data.get("columns", [])
        assert len(sample_data.get("preview", [])) > 0
        target_col = sample_data.get("suggested_target", "churn")
        print(f"[OK] Loaded sample dataset: {sample_data.get('filename')} ({sample_data.get('rows')} rows, Target: {target_col})")

        # 4. Test POST /api/analyze
        print("\n--- 4. Testing POST /api/analyze ---")
        analyze_payload = {
            "target": target_col,
            "mode": "classification"
        }
        status, headers, body = http_post_json(f"{base_url}/api/analyze", analyze_payload)
        assert status == 200, f"Expected 200, got {status}"
        res_json = json.loads(body.decode("utf-8"))
        assert res_json.get("status") == "success"
        assert "result" in res_json
        assert "visualizations" in res_json
        assert "ai_summary" in res_json
        assert "fix_script" in res_json
        risk_score = res_json["result"].get("risk_score")
        print(f"[OK] Analysis executed successfully! Calculated Risk Score: {risk_score}/100")

        # 5. Test POST /api/copilot/chat
        print("\n--- 5. Testing POST /api/copilot/chat ---")
        chat_payload = {"question": "What is the primary root cause of failure in this model?"}
        status, headers, body = http_post_json(f"{base_url}/api/copilot/chat", chat_payload)
        assert status == 200, f"Expected 200, got {status}"
        chat_json = json.loads(body.decode("utf-8"))
        assert "reply" in chat_json
        print(f"[OK] Copilot response: {chat_json.get('reply')[:120]}...")

        # 6. Test File Export
        print("\n--- 6. Testing File Export Endpoints ---")
        status, headers, body = http_get(f"{base_url}/download/summary.csv")
        assert status == 200
        assert "filename" in body.decode("utf-8")
        status, headers, body = http_get(f"{base_url}/download/analysis.json")
        assert status == 200
        status, headers, body = http_get(f"{base_url}/download/fix_pipeline.py")
        assert status == 200
        print("[OK] All download endpoints (summary.csv, analysis.json, fix_pipeline.py) verified")

        # 7. Test SEO Endpoints: Sitemap & Robots.txt
        print("\n--- 7. Testing Sitemap & Robots.txt Endpoints ---")
        import xml.etree.ElementTree as ET

        # GET /sitemap.xml
        status, headers, body = http_get(f"{base_url}/sitemap.xml")
        assert status == 200, f"Expected 200, got {status}"
        assert "application/xml" in headers.get("Content-Type", "") or "text/xml" in headers.get("Content-Type", "")
        root = ET.fromstring(body.decode("utf-8"))
        assert "urlset" in root.tag
        print("[OK] GET /sitemap.xml returns 200 with valid XML")

        # HEAD /sitemap.xml
        req_head = urllib.request.Request(f"{base_url}/sitemap.xml", method="HEAD")
        with urllib.request.urlopen(req_head) as resp:
            assert resp.status == 200
            assert "application/xml" in resp.headers.get("Content-Type", "") or "text/xml" in resp.headers.get("Content-Type", "")
        print("[OK] HEAD /sitemap.xml returns 200 without body")

        # GET /sitemap and /sitemap.xml/
        status, headers, _ = http_get(f"{base_url}/sitemap")
        assert status == 200
        status, headers, _ = http_get(f"{base_url}/sitemap.xml/")
        assert status == 200
        print("[OK] Sitemap route aliases (/sitemap, /sitemap.xml/) return 200")

        # GET /robots.txt
        status, headers, body = http_get(f"{base_url}/robots.txt")
        assert status == 200
        assert "text/plain" in headers.get("Content-Type", "")
        assert "Sitemap:" in body.decode("utf-8")
        assert "sitemap.xml" in body.decode("utf-8")
        print("[OK] GET /robots.txt returns 200 and points to sitemap.xml")

        # HEAD /robots.txt
        req_head_robots = urllib.request.Request(f"{base_url}/robots.txt", method="HEAD")
        with urllib.request.urlopen(req_head_robots) as resp:
            assert resp.status == 200
        print("[OK] HEAD /robots.txt returns 200")

        print("\n" + "=" * 50)
        print(" [OK] ALL INTEGRATION TESTS PASSED 100% PERFECTLY!")
        print("=" * 50)

    finally:
        server.shutdown()
        server.server_close()
        print("[*] Server shutdown cleanly.")

if __name__ == "__main__":
    run_tests()
