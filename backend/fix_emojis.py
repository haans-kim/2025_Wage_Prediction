import os
import glob

# Emoji replacements
replacements = {
    '✅': '[OK]',
    '⚠️': '[WARNING]',
    'ℹ️': '[INFO]',
    '❌': '[ERROR]',
    '🔍': '[SEARCH]',
    '📊': '[DATA]',
    '🗑️': '[CLEANUP]'
}

# Find all Python files
python_files = glob.glob('app/**/*.py', recursive=True)

for filepath in python_files:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        modified = content
        for emoji, replacement in replacements.items():
            modified = modified.replace(emoji, replacement)

        if modified != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(modified)
            print(f"Fixed: {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

print("Done!")
