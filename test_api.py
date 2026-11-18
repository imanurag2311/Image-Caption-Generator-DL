"""
test_api.py
===========
Comprehensive verification script for the Image Caption Generator Flask API.
Tests all endpoints, model inference (greedy and beam search), sample images,
health checks, and error handling.
"""

import sys
import os
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.server import app

def run_tests():
    client = app.test_client()
    passed = 0
    failed = 0

    def assert_true(condition, test_name):
        nonlocal passed, failed
        if condition:
            print(f"  [PASS] {test_name}")
            passed += 1
        else:
            print(f"  [FAIL] {test_name}")
            failed += 1

    print("=" * 60)
    print("IMAGE CAPTION GENERATOR - BACKEND API TEST SUITE")
    print("=" * 60)

    # 1. Health check endpoint
    print("\n1. Testing GET /api/health...")
    res = client.get("/api/health")
    assert_true(res.status_code == 200, "Health check returns 200")
    health = res.get_json()
    assert_true(health.get("status") == "healthy", "Health status is 'healthy'")
    assert_true(health.get("models_loaded") is True, "Models loaded is True")
    print(f"     Models: {health.get('models')}")

    # 2. Model info endpoint
    print("\n2. Testing GET /api/model/info...")
    res = client.get("/api/model/info")
    assert_true(res.status_code == 200, "Model info returns 200")
    info = res.get_json()
    assert_true(info.get("vocab_size") == 7000, "Vocabulary size is 7000")
    assert_true(info.get("max_length") == 35, "Max sequence length is 35")
    assert_true("evaluation_results" in info, "Evaluation results key present")
    assert_true(info.get("architecture", {}).get("total_params") is not None, "Total params present")
    print(f"     Vocab size: {info.get('vocab_size')}, Max length: {info.get('max_length')}")
    print(f"     BLEU-1: {info.get('evaluation_results', {}).get('bleu1'):.4f}")

    # 3. Sample images endpoint
    print("\n3. Testing GET /api/sample-images...")
    res = client.get("/api/sample-images")
    assert_true(res.status_code == 200, "Sample images returns 200")
    sample_data = res.get_json()
    images = sample_data.get("images", [])
    assert_true(len(images) > 0, f"Returns sample images (got {len(images)})")
    assert_true(sample_data.get("total_available", 0) > 8000, "Total available Flickr8k images > 8000")

    # 4. Serve individual sample image
    sample_img_bytes = None
    sample_img_name = images[0] if images else None
    if sample_img_name:
        print(f"\n4. Testing GET /api/sample-images/{sample_img_name}...")
        res = client.get(f"/api/sample-images/{sample_img_name}")
        assert_true(res.status_code == 200, "Sample image served with 200")
        assert_true(len(res.data) > 1000, f"Image data received ({len(res.data):,} bytes)")
        sample_img_bytes = res.data

    # 5. Captioning with Greedy search
    if sample_img_bytes:
        print("\n5. Testing POST /api/caption (Greedy Search)...")
        data = {
            "image": (io.BytesIO(sample_img_bytes), sample_img_name),
            "method": "greedy"
        }
        res = client.post("/api/caption", data=data, content_type="multipart/form-data")
        assert_true(res.status_code == 200, "Caption endpoint returns 200")
        caption_res = res.get_json()
        assert_true("caption" in caption_res and len(caption_res["caption"]) > 0, "Caption generated")
        assert_true("objects" in caption_res, "Objects detected")
        assert_true(caption_res.get("method") == "greedy", "Method is greedy")
        print(f"     Caption: \"{caption_res.get('caption')}\"")
        print(f"     Objects: {[o['class'] for o in caption_res.get('objects', [])[:5]]}")
        print(f"     Time: {caption_res.get('processing_time')}s")

        # 6. Captioning with Beam search (beam_width=2 for quick test)
        print("\n6. Testing POST /api/caption (Beam Search, width=2)...")
        data = {
            "image": (io.BytesIO(sample_img_bytes), sample_img_name),
            "method": "beam",
            "beam_width": "2"
        }
        res = client.post("/api/caption", data=data, content_type="multipart/form-data")
        assert_true(res.status_code == 200, "Beam caption returns 200")
        beam_res = res.get_json()
        assert_true("caption" in beam_res and len(beam_res["caption"]) > 0, "Beam caption generated")
        print(f"     Beam Caption: \"{beam_res.get('caption')}\"")
        print(f"     Time: {beam_res.get('processing_time')}s")

    # 7. Error handling tests
    print("\n7. Testing Error Handling...")
    # Missing image
    res = client.post("/api/caption", data={}, content_type="multipart/form-data")
    assert_true(res.status_code == 400, "Missing image returns 400")

    # Invalid method
    data = {
        "image": (io.BytesIO(b"fake image data"), "test.jpg"),
        "method": "invalid_method"
    }
    res = client.post("/api/caption", data=data, content_type="multipart/form-data")
    assert_true(res.status_code == 400, "Invalid method returns 400")

    # Invalid file extension
    data = {
        "image": (io.BytesIO(b"fake txt data"), "test.txt"),
    }
    res = client.post("/api/caption", data=data, content_type="multipart/form-data")
    assert_true(res.status_code == 400, "Disallowed file type returns 400")

    # 404 for non-existent endpoint
    res = client.get("/api/nonexistent")
    assert_true(res.status_code == 404, "Unknown endpoint returns 404")

    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed} PASSED, {failed} FAILED")
    print("=" * 60)

    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
