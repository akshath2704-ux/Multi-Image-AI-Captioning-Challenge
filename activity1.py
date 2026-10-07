from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import csv
from datetime import datetime
MODEL_ID = "Salesforce/blip-image-captioning-large" 

print(f"Loading model {MODEL_ID}...")
processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
if torch.cuda.is_available():
    model = model.to("cuda")
def get_ai_caption(image_path, custom_prompt=""):
    try:
        raw_image = Image.open(image_path).convert('RGB')
        if custom_prompt:
            text = custom_prompt
        else:
            text = "a photography of"

        inputs = processor(raw_image, text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        out = model.generate(**inputs, max_length=100)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error: {e}"

print("=== Multi-Image AI Captioning Challenge ===")
image_folder = input("Enter folder path with images (leave empty to use current folder + uploaded): ").strip()
image_files = []
uploaded = [
    "/mnt/data/wa_image_100575952248205841",
    "/mnt/data/wa_image_7647979316916557581"
]
for p in uploaded:
    if os.path.exists(p):
        image_files.append(p)
if image_folder and os.path.exists(image_folder):
    for f in os.listdir(image_folder):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_files.append(os.path.join(image_folder, f))
elif not image_folder:
    for f in os.listdir("."):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_files.append(f)

if not image_files:
    print("No images found! Please add images to folder.")
    exit()

print(f"\nFound {len(image_files)} images: {image_files}")
print("\nChoose Caption Style:")
print("1. Default (normal description)")
print("2. Detailed (ask user for context)")
print("3. Custom Prompt (you type what to focus on)")

style_choice = input("Enter 1-3 (default 1): ") or "1"

custom_instruction = ""
if style_choice == "2":
    custom_instruction = input("Enter context e.g., 'this is a vintage photo, describe damage': ")
elif style_choice == "3":
    custom_instruction = input("Enter custom prompt e.g., 'describe clothing in detail': ")
report_data = []
print("\n--- Generating Captions ---")

for idx, img_path in enumerate(image_files, 1):
    print(f"[{idx}/{len(image_files)}] Processing {os.path.basename(img_path)}...")
    caption = get_ai_caption(img_path, custom_instruction)
    print(f" -> Caption: {caption}")

    report_data.append({
        "image_id": idx,
        "image_name": os.path.basename(img_path),
        "image_path": img_path,
        "ai_caption": caption,
        "user_notes": "",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
print("\n--- Report Review (Python Fundamentals: User Input Loop) ---")
for entry in report_data:
    print(f"\nImage: {entry['image_name']}")
    print(f"AI Caption: {entry['ai_caption']}")
    user_note = input("Add your note / correction (press Enter to skip): ")
    entry["user_notes"] = user_note
csv_file = "ai_caption_report.csv"
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=["image_id", "image_name", "ai_caption", "user_notes", "timestamp"])
    writer.writeheader()
    writer.writerows(report_data)
print(f"\n✅ CSV Report saved: {csv_file}")
txt_file = "ai_caption_report.txt"
with open(txt_file, 'w', encoding='utf-8') as f:
    f.write("Multi-Image AI Captioning Report\n")
    f.write("="*40 + "\n\n")
    for entry in report_data:
        f.write(f"Image {entry['image_id']}: {entry['image_name']}\n")
        f.write(f"Caption: {entry['ai_caption']}\n")
        f.write(f"Notes: {entry['user_notes']}\n")
        f.write("-"*40 + "\n")
print(f"✅ TXT Report saved: {txt_file}")

print("\n=== Challenge Complete ===")
print("You extended basic script to:")
print("1. Handle MULTIPLE images (for loop + list)")
print("2. Track captions in a REPORT (CSV + dictionary)")
print("3. Prompt for USER INPUT (input() + enrichment loop)")