# Blitzkrieg Dynamic File Handling Updates

## Overview
Enhanced `blitzkreig.py` to be more dynamic based on the files that Blitzkrieg "revives" (loads from the TXT folder).

## Key Changes

### 1. **Dynamic File Discovery with Metadata Tracking**
- Added `loaded_files_metadata` list to track detailed information about each loaded file
- Each file now records:
  - Filename and full path
  - File size in bytes
  - Last modification timestamp
  - Character count
  - Document count
- Enhanced `load_txt()` function to capture and store this metadata

### 2. **Configurable Paths via Environment Variables**
- `BLITZKRIEG_TXT_FOLDER` - Set custom TXT folder path (default: `./agents/txts`)
- `BLITZKRIEG_DB_PATH` - Set custom FAISS database path (default: `./blitzkreig_faiss_db`)
- `BLITZKRIEG_THREADS` - Set thread count for parallel loading (default: `8`)

### 3. **Enhanced Logging and Feedback**
- Improved startup messages with formatted banners
- Detailed per-file loading feedback showing size, character count, and document count
- Warning messages when folders don't exist or are empty
- Progress tracking during ingestion and embedding

### 4. **New Built-in Tools and Commands**

#### **`blitz_show_files`** - File Summary Tool
- Shows comprehensive statistics about loaded knowledge base
- Lists all loaded files with metadata
- Can be called via: `!blitzkrieg files` or `!bk files`

#### **`blitz_reload`** - Dynamic Reload Tool  
- Reloads all .txt files from the configured folder
- Rebuilds the FAISS vectorstore index
- Can be called via: `!blitzkrieg reload` or `!bk reload`

#### **`blitz_ingest_txts`** - Enhanced Ingestion Tool (updated)
- Now includes detailed logging at each step
- Shows chunk statistics after splitting
- Returns comprehensive summary including file metadata

### 5. **Help Command**
- New `!blitzkrieg help` or `!bk help` command
- Shows all available commands and tools
- Dynamically lists loaded external tools

### 6. **Improved Discord Message Handling**
- Automatically splits long messages to comply with Discord's 2000 character limit
- Better error handling for tool execution
- More informative responses

## Usage Examples

### Check Loaded Files
```
!blitzkrieg files
```
or
```
!bk files
```

Response:
```
📚 **Blitzkrieg Knowledge Base Summary**
Total files: 5
Total documents: 23
Total characters: 45,678
Total size: 52,341 bytes

**Loaded Files:**
  • document1.txt: 8,234 chars, 3 docs (modified: 2025-12-27 10:30:45)
  • document2.txt: 12,567 chars, 6 docs (modified: 2025-12-27 11:15:22)
  ...
```

### Reload Knowledge Base
```
!blitzkrieg reload
```
or
```
!bk reload
```

This will:
1. Scan the TXT folder for all .txt files
2. Load and parse each file
3. Split documents into chunks
4. Generate embeddings
5. Rebuild the FAISS index
6. Return a detailed summary of the operation

### View Help
```
!blitzkrieg help
```
or
```
!bk help
```

## Technical Improvements

### File Loading
- Better error handling for missing folders and encoding issues
- Parallel file loading using ThreadPoolExecutor
- Metadata collection for monitoring and debugging

### Vectorstore Management
- Configurable chunk size (1000) and overlap (100)
- Progress logging during embedding generation
- Automatic index persistence

### Tool System
- More organized tool registration
- Built-in tools now properly documented
- External tool loading with better error reporting

## Benefits

1. **Transparency**: Users can see exactly what files are loaded and when
2. **Flexibility**: Environment variables allow customization without code changes
3. **Debugging**: Detailed metadata helps troubleshoot loading issues
4. **Dynamic Updates**: Reload command allows updating knowledge base without restarting the bot
5. **User Experience**: Clear feedback and help commands improve usability

## Environment Variables

Add these to your `.env` file:

```bash
# Discord
DISCORD_TOKEN=your_discord_token_here

# Blitzkrieg Configuration (optional, defaults shown)
BLITZKRIEG_TXT_FOLDER=./agents/txts
BLITZKRIEG_DB_PATH=./blitzkreig_faiss_db
BLITZKRIEG_THREADS=8
```

## Startup Output Example

```
🚀 Blitzkrieg Bot Starting Up...
📂 TXT Folder: ./agents/txts
💾 Database Path: ./blitzkreig_faiss_db
🧵 Thread Count: 8

======================================================================
[📚 Blitzkreig Knowledge Base Loading Results]
======================================================================
✅ Loaded: doc1.txt (5234 bytes, 4892 chars, 2 docs)
✅ Loaded: doc2.txt (8901 bytes, 8234 chars, 4 docs)
⚠️ Skipped empty.txt: empty
✅ Loaded: doc3.txt (3456 bytes, 3210 chars, 1 docs)

📊 Total loaded documents: 7
📁 Files loaded: 3
📝 Total characters: 16,336
======================================================================
```
