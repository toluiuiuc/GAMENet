import os

saved_model_path = "saved/GAMENet"
files = os.listdir(saved_model_path)
for f in files:
    if "Epoch" in f:
        resume_path = os.path.join(saved_model_path, f)
        print(resume_path)
        os.system(
            f"python train_GAMENet.py --model_name GAMENet --ddi --resume_path {resume_path} --eval --remove_dm remove_input"
        )
