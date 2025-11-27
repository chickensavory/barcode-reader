# Barcode Changer
# Wait! May issue toh! Missing the main() function
This tool automatically **reads barcodes from images and renames the files for you**.  
Everything below is written so you can simply **copy, paste, and follow along**.

---

# 🌟 What This Tool Does

When you run the command:

- It looks for a folder named **input**
- It reads your image files **two at a time**
- It detects and reads the **barcodes**
- It renames the images automatically based on the barcode values
---

# 💻 What You Need

- A Mac  
- Python 3 installed (most Macs already have it)

You **do NOT** need:

- Admin or root access  
- Coding or technical knowledge  
- To install any apps  
- To manage complicated settings  

---

# 📦 Installation (No Admin Needed)

Follow these steps exactly:

---

## 1. Open Terminal

Press:

**⌘ + Space → type “Terminal” → press Enter**

---

## 2. Install the tool (copy and paste this)

```bash
pip install --user "git+https://github.com/chickensavory/barcode-changer.git"
```

This installs the tool **for your user only** — safe and simple.

---

## 3. Close Terminal, then reopen it

This step is important so the command becomes available.

---

# 📸 How To Use the Tool (Step-by-Step)

---

## Step 1 — Create a working folder

You can name it anything. Example:

```bash
mkdir barcode_work
cd barcode_work
```

---

## Step 2 — Make an `input` folder inside it

```bash
mkdir input
```

This `input` folder must contain the images you want processed.

---

## Step 3 — Add your image files

Using Finder:

1. Open your `barcode_work` folder  
2. Drag your images into the `input` folder

No special formatting required — just put your image pairs in there.

---

## Step 4 — Run the tool

Make sure you are still in your working folder (the one **containing** the `input` folder), then run:

```bash
barcode-changer
```

---

## Step 5 — Watch it work!

The tool will:

- Detect barcodes  
- Process your images in pairs  
- Rename them  
- Print progress in Terminal  

Your renamed files will appear in the same folder.

---

# 🧪 Complete Example (From Scratch)

Let’s pretend you are starting fresh.

1. Create a folder on your Desktop named **photos**  
2. Inside **photos**, create a folder named **input**  
3. Drag all your RAW/JPG images into **input**  
4. Open Terminal and run:

```bash
cd ~/Desktop/photos
barcode-changer
```

5. The tool renames your files based on detected barcodes  
6. You’re done 🎉

---

# 🔄 Updating the Tool

If a new version is published later, update easily:

```bash
pip install --user --upgrade "git+https://github.com/chickensavory/barcode-changer.git"
```

---

# ❗ Troubleshooting & Fixes

---

### Problem: “command not found: barcode-changer”

Try this:

1. Close Terminal  
2. Reopen Terminal  
3. Try running `barcode-changer` again

If still failing, run:

```bash
python3 -m pip install --user "git+https://github.com/chickensavory/barcode-changer.git"
```

---

### Problem: “pip: command not found”

Run:

```bash
python3 -m ensurepip --upgrade
```

Then try the installation command again.

---

### Problem: You don’t have Python 3

Download it here (official site):

https://www.python.org/downloads/macos/

Restart Terminal after installing.

---
