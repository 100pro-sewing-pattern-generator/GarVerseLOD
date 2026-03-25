from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import subprocess
import os

app = FastAPI()

BASE_DIR = Path(__file__).parent
IMG_DIR = BASE_DIR / "inputs/imgs"
OUTPUT_DIR = BASE_DIR / "outputs/temp"

@app.post("/full_pipeline")
async def full_pipeline(file: UploadFile = File(...)):
    try:
        # 1. 入力画像保存
        IMG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = IMG_DIR / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        logs = {}

        # ----------------------------
        # 0_normal_estimator
        # ----------------------------
        result0 = subprocess.run(
            [
                "python",
                str(BASE_DIR / "0_normal_estimator/predict_normal.py"),
                "--input_dir", str(IMG_DIR),
                "--output_dir", str(OUTPUT_DIR)
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR / "0_normal_estimator")
        )
        logs["normal_estimator"] = {"stdout": result0.stdout, "stderr": result0.stderr}

        # ----------------------------
        # 1_coarse/ICON_get_smpl
        # ----------------------------
        coarse_temp_dir = OUTPUT_DIR / "coarse_temp"
        coarse_temp_dir.mkdir(parents=True, exist_ok=True)

        env_icon = os.environ.copy()
        env_icon["CUDA_VISIBLE_DEVICES"] = "0"

        result1 = subprocess.run(
            [
                "python", "-m", "apps.infer_smpl",
                "-cfg", "./configs/icon-filter.yaml",
                "-gpu", "0",
                "-in_dir", str(IMG_DIR),
                "-out_dir", str(coarse_temp_dir),
                "-export_video",
                "-loop_smpl", "1",
                "-loop_cloth", "200",
                "-hps_type", "pymaf"
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR / "1_coarse/ICON_get_smpl"),
            env=env_icon
        )
        logs["infer_smpl"] = {"stdout": result1.stdout, "stderr": result1.stderr}

        # ----------------------------
        # 1_coarse/tpose_garment_estimator
        # ----------------------------
        coarse_temp_dir.mkdir(parents=True, exist_ok=True)
        env_tpose = os.environ.copy()
        env_tpose["CUDA_VISIBLE_DEVICES"] = "0"

        result2 = subprocess.run(
            [
                "python", "test_wild.py",
                "--in_folder", str(IMG_DIR),
                "--out_folder", str(coarse_temp_dir)
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR / "1_coarse/tpose_garment_estimator"),
            env=env_tpose
        )
        logs["tpose_garment"] = {"stdout": result2.stdout, "stderr": result2.stderr}

        # ----------------------------
        # pose_garment.py
        # ----------------------------
        coarse_garment_dir = OUTPUT_DIR / "coarse_garment"
        coarse_garment_dir.mkdir(parents=True, exist_ok=True)

        env_pose = os.environ.copy()
        env_pose["CUDA_VISIBLE_DEVICES"] = "0"

        result3 = subprocess.run(
            [
                "python", "pose_garment.py",
                "--in_folder", str(IMG_DIR),
                "--out_folder", str(coarse_garment_dir),
                "--temp", str(coarse_temp_dir)
            ],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR / "1_coarse/smpl_lbs_to_garment"),
            env=env_pose
        )
        logs["pose_garment"] = {"stdout": result3.stdout, "stderr": result3.stderr}

        return JSONResponse(content={
            "status": "success",
            "logs": logs,
            "file_saved": str(file_path),
            "coarse_temp_dir": str(coarse_temp_dir),
            "coarse_garment_dir": str(coarse_garment_dir)
        })

    except subprocess.CalledProcessError as e:
        print(e)
        return JSONResponse(content={
            "status": "error",
            "file_saved": str(file_path),
            "script_stdout": e.stdout,
            "script_stderr": e.stderr
        }, status_code=500)