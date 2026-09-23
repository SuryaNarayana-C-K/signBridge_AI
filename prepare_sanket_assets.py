import os
import sys
import json
import math
import struct
import zlib
import urllib.request
import urllib.parse
import http.server
import threading

KTIME = 46186158000.0

# 1. Download Model & Textures
base_url = "https://raw.githubusercontent.com/krishnshyam/VirtualISLInterpreter/main/anim/model-Sanket/"
out_dir = r"c:\Users\SURYA NARAYAN C K\Desktop\hackkkkkkkkkkkkkk\sanket"
os.makedirs(out_dir, exist_ok=True)
tex_dir = os.path.join(out_dir, "textures")
os.makedirs(tex_dir, exist_ok=True)
signs_dir = os.path.join(out_dir, "signs")
os.makedirs(signs_dir, exist_ok=True)

files = [
    ("MaleModelSankit.fbx", os.path.join(out_dir, "MaleModelSankit.fbx")),
    ("textures/ArmaniHairCapTR - Copy.jpg", os.path.join(tex_dir, "ArmaniHairCapTR - Copy.jpg")),
    ("textures/Shirt-A.jpeg", os.path.join(tex_dir, "Shirt-A.jpeg")),
    ("textures/ablackhair.jpg", os.path.join(tex_dir, "ablackhair.jpg")),
    ("textures/human_diffusedit.jpg", os.path.join(tex_dir, "human_diffusedit.jpg")),
    ("textures/old_darkskinned_male_diffuse.png", os.path.join(tex_dir, "old_darkskinned_male_diffuse.png")),
]

for rel_path, dest_path in files:
    if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
        url = base_url + urllib.parse.quote(rel_path)
        try:
            urllib.request.urlretrieve(url, dest_path)
            print(f"Downloaded {rel_path} ({os.path.getsize(dest_path)} bytes)")
        except Exception as e:
            print(f"Failed to download {rel_path}: {e}")

# 2. Download Real ISL FBX Animation Clips
target_signs = [
    ("please", "https://raw.githubusercontent.com/krishnshyam/VirtualISLInterpreter/main/anim/signs/fbx/please.fbx"),
    ("come", "https://raw.githubusercontent.com/krishnshyam/VirtualISLInterpreter/main/anim/signs/fbx/come%201.fbx"),
    ("go", "https://raw.githubusercontent.com/krishnshyam/VirtualISLInterpreter/main/anim/signs/fbx/go.fbx"),
    ("like", "https://raw.githubusercontent.com/krishnshyam/VirtualISLInterpreter/main/anim/signs/fbx/like.fbx"),
    ("home", "https://raw.githubusercontent.com/krishnshyam/VirtualISLInterpreter/main/anim/signs/fbx/home.fbx"),
    ("work", "https://raw.githubusercontent.com/krishnshyam/VirtualISLInterpreter/main/anim/signs/fbx/work.fbx"),
    ("car", "https://raw.githubusercontent.com/krishnshyam/VirtualISLInterpreter/main/anim/signs/fbx/car.fbx"),
    ("drive", "https://raw.githubusercontent.com/krishnshyam/VirtualISLInterpreter/main/anim/signs/fbx/drive.fbx"),
    ("action", "https://raw.githubusercontent.com/krishnshyam/VirtualISLInterpreter/main/anim/signs/fbx/action.fbx"),
    ("before", "https://raw.githubusercontent.com/krishnshyam/VirtualISLInterpreter/main/anim/signs/fbx/before%201.fbx"),
]

for sign_name, sign_url in target_signs:
    fbx_path = os.path.join(signs_dir, f"{sign_name}.fbx")
    if not os.path.exists(fbx_path) or os.path.getsize(fbx_path) == 0:
        try:
            urllib.request.urlretrieve(sign_url, fbx_path)
            print(f"Downloaded sign '{sign_name}' ({os.path.getsize(fbx_path)} bytes)")
        except Exception as e:
            print(f"Failed to download sign '{sign_name}': {e}")

print("All asset downloads completed successfully!")
