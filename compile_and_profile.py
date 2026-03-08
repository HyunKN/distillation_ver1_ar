"""
QAI Hub Compile & Profile Script for LPCVC CLIP-Lite
- Compiles image and text encoders to QNN
- Profiles on target device (XR2 Gen 2 - LPCVC 2026 Track 1 target)
"""
import argparse
import qai_hub
import onnx
import os
import sys

def compile_model(model, device, input_specs, name):
    """Submit a compile job and wait for completion."""
    print(f"  Submitting compile job for {name}...")
    compile_job = qai_hub.submit_compile_job(
        model=model,
        device=device,
        input_specs=input_specs,
        options="--target_runtime qnn_dlc --truncate_64bit_io"
    )
    print(f"  Compile job ID: {compile_job.job_id}")
    print(f"  Waiting for compilation to complete...")
    compile_job.wait()
    print(f"  ✅ {name} compilation complete!")
    return compile_job

def run_profile(compiled_model, device, name):
    """Submit a profile job for the compiled model."""
    print(f"  Submitting profile job for {name}...")
    profile_job = qai_hub.submit_profile_job(
        model=compiled_model,
        device=device,
        options="--max_profiler_iterations 100"
    )
    print(f"  Profile job ID: {profile_job.job_id}")
    return profile_job

def infer_input_spec(onnx_model):
    """Infer a static QAI Hub input spec from an ONNX model."""
    tensor = onnx_model.graph.input[0]
    shape = []
    for dim in tensor.type.tensor_type.shape.dim:
        if dim.dim_value > 0:
            shape.append(int(dim.dim_value))
        else:
            shape.append(1)
    elem_type = tensor.type.tensor_type.elem_type
    if elem_type == onnx.TensorProto.INT32:
        return {tensor.name: (tuple(shape), "int32")}
    return {tensor.name: tuple(shape)}

def main():
    ap = argparse.ArgumentParser(description="Compile and profile ONNX models on QAI Hub")
    ap.add_argument("--onnx_dir", default="exported_onnx", help="Directory containing ONNX files")
    ap.add_argument("--img_name", default="image_encoder.onnx", help="Image encoder ONNX filename")
    ap.add_argument("--txt_name", default="text_encoder.onnx", help="Text encoder ONNX filename")
    ap.add_argument("--device", default="XR2 Gen 2 (Proxy)", help="Target device name (LPCVC 2026: XR platform)")
    ap.add_argument("--skip_profile", action="store_true", help="Skip profiling step")
    args = ap.parse_args()

    # Construct full paths
    img_path = os.path.join(args.onnx_dir, args.img_name)
    txt_path = os.path.join(args.onnx_dir, args.txt_name)

    # Check files exist
    if not os.path.exists(img_path):
        print(f"❌ Error: Image ONNX not found: {img_path}")
        sys.exit(1)
    if not os.path.exists(txt_path):
        print(f"❌ Error: Text ONNX not found: {txt_path}")
        sys.exit(1)

    # Load and validate ONNX models
    print(f"\n📦 Loading ONNX models...")
    print(f"  Image: {img_path}")
    onnx_img = onnx.load(img_path)
    try:
        onnx.checker.check_model(onnx_img)
        print(f"  ✅ Image ONNX is valid")
    except onnx.checker.ValidationError as e:
        print(f"  ❌ Image ONNX validation failed: {e}")
        sys.exit(1)

    print(f"  Text:  {txt_path}")
    onnx_txt = onnx.load(txt_path)
    try:
        onnx.checker.check_model(onnx_txt)
        print(f"  ✅ Text ONNX is valid")
    except onnx.checker.ValidationError as e:
        print(f"  ❌ Text ONNX validation failed: {e}")
        sys.exit(1)

    # Target device
    device = qai_hub.Device(args.device)
    print(f"\n🎯 Target device: {args.device}")
    img_input_spec = infer_input_spec(onnx_img)
    txt_input_spec = infer_input_spec(onnx_txt)

    # Compile models
    print(f"\n🔧 Compiling models...")
    img_compile_job = compile_model(
        model=onnx_img,
        device=device,
        input_specs=img_input_spec,
        name="Image Encoder"
    )
    txt_compile_job = compile_model(
        model=onnx_txt,
        device=device,
        input_specs=txt_input_spec,
        name="Text Encoder"
    )

    # Get compiled models
    compiled_img = img_compile_job.get_target_model()
    compiled_txt = txt_compile_job.get_target_model()

    print(f"\n✅ Compilation Summary:")
    print(f"  Image Encoder Job: {img_compile_job.job_id}")
    print(f"  Text Encoder Job:  {txt_compile_job.job_id}")

    # Profile models (optional)
    if not args.skip_profile:
        print(f"\n📊 Profiling models...")
        img_profile_job = run_profile(compiled_img, device, "Image Encoder")
        txt_profile_job = run_profile(compiled_txt, device, "Text Encoder")

        print(f"\n📊 Profiling Summary:")
        print(f"  Image Encoder Profile Job: {img_profile_job.job_id}")
        print(f"  Text Encoder Profile Job:  {txt_profile_job.job_id}")
        print(f"\n💡 Check results at: https://aihub.qualcomm.com/jobs")
    else:
        print(f"\n⏭️ Profiling skipped (--skip_profile)")

    print(f"\n🎉 Done!")

if __name__ == "__main__":
    main()
