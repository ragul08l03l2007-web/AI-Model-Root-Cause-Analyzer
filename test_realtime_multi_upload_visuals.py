# test_realtime_multi_upload_visuals.py
import io
import json
import re
import urllib.request
import urllib.parse
from http.server import ThreadingHTTPServer
import threading
import time
import pandas as pd
import numpy as np

from app import AppHandler, reset_state
from analysis.visualization import generate_all_visualizations


def run_tests():
    print("=" * 60)
    print("TESTING REAL-TIME DYNAMIC VISUAL GRAPHS ACROSS MULTIPLE UPLOADS")
    print("=" * 60)

    reset_state()
    server = ThreadingHTTPServer(("localhost", 8082), AppHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.5)

    base_url = "http://localhost:8082"

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = urllib.request.build_opener(NoRedirect)

    def post_no_redirect(req):
        try:
            resp = opener.open(req)
            return resp.status, resp.headers.get("Location")
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Location")

    try:
        # -------------------------------------------------------------
        # 1. UPLOAD DATASET A: Binary Classification (Churn / Credit)
        # -------------------------------------------------------------
        print("\n[STEP 1] Uploading and Analyzing Dataset A (Binary Classification)...")
        csv_a = (
            "user_id,tenure,monthly_charges,total_spend,contract_type,churn\n"
            "1,12,70.5,846.0,month-to-month,1\n"
            "2,48,110.0,5280.0,two-year,0\n"
            "3,3,45.0,135.0,month-to-month,1\n"
            "4,24,80.0,1920.0,one-year,0\n"
            "5,6,90.0,540.0,month-to-month,1\n"
            "6,60,115.0,6900.0,two-year,0\n"
            "7,1,55.0,55.0,month-to-month,1\n"
            "8,36,95.0,3420.0,one-year,0\n"
            "9,18,65.0,1170.0,month-to-month,0\n"
            "10,72,120.0,8640.0,two-year,0\n"
            "11,2,50.0,100.0,month-to-month,1\n"
            "12,30,85.0,2550.0,one-year,0\n"
            "13,10,75.0,750.0,month-to-month,1\n"
            "14,50,105.0,5250.0,two-year,0\n"
            "15,8,60.0,480.0,month-to-month,1\n"
            "16,40,90.0,3600.0,one-year,0\n"
            "17,5,85.0,425.0,month-to-month,1\n"
            "18,65,110.0,7150.0,two-year,0\n"
            "19,15,70.0,1050.0,month-to-month,0\n"
            "20,22,80.0,1760.0,one-year,0\n"
        ).encode("utf-8")

        boundary = "----WebKitFormBoundaryDatasetA"
        body_a = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="mode"\r\n\r\n'
            f"classification\r\n"
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="dataset"; filename="telecom_churn.csv"\r\n'
            f"Content-Type: text/csv\r\n\r\n"
        ).encode("utf-8") + csv_a + f"\r\n--{boundary}--\r\n".encode("utf-8")

        upload_req = urllib.request.Request(
            f"{base_url}/upload",
            data=body_a,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
        )
        status, loc = post_no_redirect(upload_req)
        assert status == 302 and loc == "/preview", f"Expected 302 -> /preview, got {status} -> {loc}"

        # Analyze on 'churn'
        form_data = urllib.parse.urlencode({"target": "churn", "mode": "classification"}).encode("utf-8")
        analyze_req = urllib.request.Request(
            f"{base_url}/analyze",
            data=form_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        status, loc = post_no_redirect(analyze_req)
        assert status == 302 and loc == "/dashboard", f"Expected 302 -> /dashboard, got {status} -> {loc}"

        dash_resp = urllib.request.urlopen(f"{base_url}/dashboard")
        assert dash_resp.status == 200
        dash_html_a = dash_resp.read().decode("utf-8")

        # Extract DIAGNOSTIC_CHARTS json embedded in HTML
        match_a = re.search(r"var DIAGNOSTIC_CHARTS = (\{.*?\});", dash_html_a, re.DOTALL)
        assert match_a is not None, "DIAGNOSTIC_CHARTS not found in dashboard HTML A!"
        charts_a = json.loads(match_a.group(1))

        print(f"  -> Dataset A Charts rendered: {list(charts_a.keys())}")
        assert "chart-perf-metrics" in charts_a
        assert "chart-target-dist" in charts_a
        assert "chart-feat-importance" in charts_a
        assert "chart-confusion-matrix" in charts_a

        feat_chart_a = charts_a["chart-feat-importance"]
        feat_y_a = feat_chart_a["data"][0]["y"]
        print(f"  -> Dataset A Feature Impact Features: {feat_y_a}")
        assert any("tenure" in f or "monthly_charges" in f or "contract_type" in f for f in feat_y_a)
        assert not any("engine_size" in f or "house_sqft" in f for f in feat_y_a)

        # -------------------------------------------------------------
        # 2. UPLOAD DATASET B: Continuous Regression (Housing / Pricing)
        # -------------------------------------------------------------
        print("\n[STEP 2] Uploading and Analyzing Dataset B (Continuous Regression)...")
        csv_b = (
            "property_id,house_sqft,bedrooms,lot_size,neighborhood_score,price\n"
            "101,1200,2,4000,7.5,250000\n"
            "102,2400,4,8000,8.8,510000\n"
            "103,950,1,3000,6.0,180000\n"
            "104,1800,3,6000,8.0,380000\n"
            "105,3200,5,10000,9.2,720000\n"
            "106,1400,2,4500,7.0,290000\n"
            "107,2100,3,7000,8.5,440000\n"
            "108,1600,3,5000,7.8,340000\n"
            "109,2800,4,9000,9.0,610000\n"
            "110,1100,2,3500,6.5,220000\n"
            "111,2600,4,8500,8.9,560000\n"
            "112,1300,2,4200,7.2,270000\n"
            "113,1950,3,6500,8.2,410000\n"
            "114,3500,5,12000,9.5,800000\n"
            "115,1750,3,5500,7.9,370000\n"
            "116,2250,4,7500,8.6,480000\n"
            "117,1050,1,3200,6.2,200000\n"
            "118,3000,4,9500,9.1,670000\n"
            "119,1500,3,4800,7.4,310000\n"
            "120,2000,3,6800,8.3,425000\n"
        ).encode("utf-8")

        boundary_b = "----WebKitFormBoundaryDatasetB"
        body_b = (
            f"--{boundary_b}\r\n"
            f'Content-Disposition: form-data; name="mode"\r\n\r\n'
            f"regression\r\n"
            f"--{boundary_b}\r\n"
            f'Content-Disposition: form-data; name="dataset"; filename="housing_pricing.csv"\r\n'
            f"Content-Type: text/csv\r\n\r\n"
        ).encode("utf-8") + csv_b + f"\r\n--{boundary_b}--\r\n".encode("utf-8")

        upload_req_b = urllib.request.Request(
            f"{base_url}/upload",
            data=body_b,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary_b}"}
        )
        status_b, loc_b = post_no_redirect(upload_req_b)
        assert status_b == 302 and loc_b == "/preview", f"Expected 302 -> /preview, got {status_b} -> {loc_b}"

        # Analyze on 'price'
        form_data_b = urllib.parse.urlencode({"target": "price", "mode": "regression"}).encode("utf-8")
        analyze_req_b = urllib.request.Request(
            f"{base_url}/analyze",
            data=form_data_b,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        status_b, loc_b = post_no_redirect(analyze_req_b)
        assert status_b == 302 and loc_b == "/dashboard", f"Expected 302 -> /dashboard, got {status_b} -> {loc_b}"

        dash_resp_b = urllib.request.urlopen(f"{base_url}/dashboard")
        assert dash_resp_b.status == 200
        dash_html_b = dash_resp_b.read().decode("utf-8")

        # Extract DIAGNOSTIC_CHARTS json embedded in HTML
        match_b = re.search(r"var DIAGNOSTIC_CHARTS = (\{.*?\});", dash_html_b, re.DOTALL)
        assert match_b is not None, "DIAGNOSTIC_CHARTS not found in dashboard HTML B!"
        charts_b = json.loads(match_b.group(1))

        print(f"  -> Dataset B Charts rendered: {list(charts_b.keys())}")
        # In Regression: actual_vs_predicted, residual_plot, residual_dist should be present
        assert "chart-actual-vs-pred" in charts_b
        assert "chart-residual-plot" in charts_b
        assert "chart-residual-dist" in charts_b
        assert "chart-confusion-matrix" not in charts_b

        feat_chart_b = charts_b["chart-feat-importance"]
        feat_y_b = feat_chart_b["data"][0]["y"]
        print(f"  -> Dataset B Feature Impact Features: {feat_y_b}")
        assert any("house_sqft" in f or "lot_size" in f or "neighborhood_score" in f for f in feat_y_b)
        # CRITICAL VERIFICATION: No features or values from Dataset A should leak into Dataset B's graphs!
        assert not any("tenure" in f or "monthly_charges" in f or "contract_type" in f for f in feat_y_b)

        # -------------------------------------------------------------
        # 3. UPLOAD DATASET C: 3-Class Multiclass (Customer Tier)
        # -------------------------------------------------------------
        print("\n[STEP 3] Uploading and Analyzing Dataset C (3-Class Multiclass)...")
        csv_c = (
            "member_id,annual_revenue,support_calls,account_age_years,tier\n"
            "1,25000,1,1,Bronze\n"
            "2,75000,4,3,Silver\n"
            "3,150000,10,5,Gold\n"
            "4,30000,2,1,Bronze\n"
            "5,80000,5,2,Silver\n"
            "6,180000,8,6,Gold\n"
            "7,28000,1,1,Bronze\n"
            "8,85000,3,4,Silver\n"
            "9,160000,9,5,Gold\n"
            "10,22000,0,1,Bronze\n"
            "11,90000,4,3,Silver\n"
            "12,200000,12,7,Gold\n"
            "13,32000,2,2,Bronze\n"
            "14,70000,3,3,Silver\n"
            "15,170000,7,5,Gold\n"
            "16,29000,1,1,Bronze\n"
            "17,95000,6,4,Silver\n"
            "18,190000,11,6,Gold\n"
        ).encode("utf-8")

        boundary_c = "----WebKitFormBoundaryDatasetC"
        body_c = (
            f"--{boundary_c}\r\n"
            f'Content-Disposition: form-data; name="mode"\r\n\r\n'
            f"classification\r\n"
            f"--{boundary_c}\r\n"
            f'Content-Disposition: form-data; name="dataset"; filename="customer_tier.csv"\r\n'
            f"Content-Type: text/csv\r\n\r\n"
        ).encode("utf-8") + csv_c + f"\r\n--{boundary_c}--\r\n".encode("utf-8")

        upload_req_c = urllib.request.Request(
            f"{base_url}/upload",
            data=body_c,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary_c}"}
        )
        status_c, loc_c = post_no_redirect(upload_req_c)
        assert status_c == 302 and loc_c == "/preview", f"Expected 302 -> /preview, got {status_c} -> {loc_c}"

        # Analyze on 'tier'
        form_data_c = urllib.parse.urlencode({"target": "tier", "mode": "classification"}).encode("utf-8")
        analyze_req_c = urllib.request.Request(
            f"{base_url}/analyze",
            data=form_data_c,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        status_c, loc_c = post_no_redirect(analyze_req_c)
        assert status_c == 302 and loc_c == "/dashboard", f"Expected 302 -> /dashboard, got {status_c} -> {loc_c}"

        dash_resp_c = urllib.request.urlopen(f"{base_url}/dashboard")
        assert dash_resp_c.status == 200
        dash_html_c = dash_resp_c.read().decode("utf-8")

        match_c = re.search(r"var DIAGNOSTIC_CHARTS = (\{.*?\});", dash_html_c, re.DOTALL)
        assert match_c is not None, "DIAGNOSTIC_CHARTS not found in dashboard HTML C!"
        charts_c = json.loads(match_c.group(1))

        print(f"  -> Dataset C Charts rendered: {list(charts_c.keys())}")
        feat_chart_c = charts_c["chart-feat-importance"]
        feat_y_c = feat_chart_c["data"][0]["y"]
        print(f"  -> Dataset C Feature Impact Features: {feat_y_c}")
        assert any("annual_revenue" in f or "support_calls" in f or "account_age_years" in f for f in feat_y_c)
        # Verify 3-class target distribution
        target_chart_c = charts_c["chart-target-dist"]
        labels_c = target_chart_c["data"][0]["labels"]
        print(f"  -> Dataset C Target Classes: {labels_c}")
        assert "Bronze" in labels_c and "Silver" in labels_c and "Gold" in labels_c

        print("\n=================================================================")
        print("ALL REAL-TIME MULTI-DATASET SEQUENTIAL UPLOAD TESTS PASSED (3/3)!")
        print("=================================================================\n")

    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    run_tests()
