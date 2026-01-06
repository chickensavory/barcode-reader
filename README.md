# Barcode Reader

This tool **reads barcodes from image files and renames the images for you**.  
Everything below is meant to be **copy → paste → run** friendly (no admin needed).

> **Command name:** This repo installs the command **`barcode-changer`**.

---

##  What It Does

When you run the command, it:

- Looks for a folder named **`input`**
- Reads your image files **two at a time** (pairs)
- Detects and reads the **barcodes**
- Renames the images based on the barcode values
- Prints progress in Terminal as it goes

---

## What You Need

- A Mac
- **Python 3**

You do **not** need:
- Admin/root access
- Coding experience
- Any extra apps

---

## Installation

### 1) Open Terminal
Press:

**⌘ + Space → type “Terminal” → press Enter**

### 2) Install the tool (copy + paste)

> Note: On many Macs `pip` isn’t available as a command, so we use `python3 -m pip`.

```bash
python3 -m pip install --user --upgrade pip
python3 -m pip install --no-binary pyrxing "git+https://github.com/chickensavory/barcode-reader.git"
````

### 3) Verify it installed (run the command)

```bash
barcode-changer
```

If you see an error like `command not found: barcode-changer`, do the one-time setup below.

### 4) One-time setup (only if the command is “not found”)

Sometimes macOS doesn’t automatically look in the folder where Python installs commands.

Run:

```bash
python3 -m site --user-base
```

If it prints something like `/Users/YOURNAME/Library/Python/3.9`, then run:

```bash
echo 'export PATH="$HOME/Library/Python/3.9/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
hash -r
```

> If your path shows a different version (example: `3.10` or `3.11`), replace `3.9` above to match.

Now try again:

```bash
barcode-changer
```

---

## How To Use (Step-by-Step)

### Step 1 — Make a working folder

You can name it anything:

```bash
mkdir barcode_work
cd barcode_work
```

### Step 2 — Create an `input` folder

```bash
mkdir input
```

### Step 3 — Add your image files

Using Finder:

1. Open your `barcode_work` folder
2. Drag your images into the `input` folder

Put images in the order you want processed—this tool reads them in pairs.

### Step 4 — Run the tool

From inside the folder that contains `input`, run:

```bash
barcode-changer
```

### Step 5 — Done

Your files will be renamed and you’ll see status output in Terminal.

---

## Updating

```bash
python3 -m pip install --user --upgrade --no-cache-dir "git+https://github.com/chickensavory/barcode-reader.git"
```

---

## 🧯 Troubleshooting

### Problem: `pip` not found

Run:

```bash
python3 -m ensurepip --upgrade
```

Then install again:

```bash
python3 -m pip install --user "git+https://github.com/chickensavory/barcode-reader.git"
```

---

### Problem: `command not found: barcode-changer`

Follow the “One-time setup” step in **Installation** to add the Python user `bin` folder to your PATH.

---

### Problem: You don’t have Python 3

Install Python 3 from the official site, then reopen Terminal:

```text
https://www.python.org/downloads/macos/
```

---

## Notes

* The tool expects a folder named **`input`** in your current directory.
* Images are processed **two at a time** (pairs).
* Keep a backup of your originals the first time you run it, just in case you want to revert.
* If the product is more than 5 in the the good folder, check if it actually is a new product (this is because this is using the shoot time as one of the reference of a new product and it might have been to close from the previous product.)