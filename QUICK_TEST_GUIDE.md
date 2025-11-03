# Quick Test Guide - VAD & Diarization Controls

## Updated: November 1, 2025

### Start the Server

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx
python3 server_production.py
```

Server will start on: **http://localhost:9004**

---

## Web Interface Testing

### 1. Open Web GUI
```
http://localhost:9004/web
```

### 2. Visual Checks
- [ ] Two checkboxes visible (VAD Filter and Diarization)
- [ ] VAD checkbox is **checked** by default
- [ ] Diarization checkbox is **unchecked** by default
- [ ] Descriptive text appears under each checkbox
- [ ] Checkboxes are 18px and have pointer cursor

### 3. Theme Testing
- [ ] Light theme: Checkboxes visible and styled correctly
- [ ] Dark theme: Text readable, checkboxes visible
- [ ] Unicorn theme: Magical gradient doesn't obscure controls

### 4. Functional Test Cases

#### Test Case 1: Default Settings (VAD ON, Diarization OFF)
1. Upload audio file
2. Click Transcribe
3. Check results:
   - VAD Filter: **✓ ON** (green)
   - Diarization: **✗ OFF** (muted)

#### Test Case 2: VAD OFF, Diarization ON
1. Uncheck VAD Filter
2. Check Diarization
3. Upload audio file
4. Click Transcribe
5. Check results:
   - VAD Filter: **✗ OFF** (muted)
   - Diarization: **✓ ON** (green)

#### Test Case 3: Both ON
1. Check both boxes
2. Upload audio file
3. Click Transcribe
4. Check results:
   - VAD Filter: **✓ ON** (green)
   - Diarization: **✓ ON** (green)

#### Test Case 4: Both OFF
1. Uncheck both boxes
2. Upload audio file
3. Click Transcribe
4. Check results:
   - VAD Filter: **✗ OFF** (muted)
   - Diarization: **✗ OFF** (muted)

---

## API Testing with curl

### Test 1: VAD ON, Diarization OFF (default)
```bash
curl -X POST \
  -F "file=@test_audio.wav" \
  -F "model=base" \
  -F "vad_filter=true" \
  -F "enable_diarization=false" \
  http://localhost:9004/transcribe
```

### Test 2: VAD OFF, Diarization ON
```bash
curl -X POST \
  -F "file=@test_audio.wav" \
  -F "model=base" \
  -F "vad_filter=false" \
  -F "enable_diarization=true" \
  http://localhost:9004/transcribe
```

### Test 3: Both enabled
```bash
curl -X POST \
  -F "file=@test_audio.wav" \
  -F "model=base" \
  -F "vad_filter=true" \
  -F "enable_diarization=true" \
  http://localhost:9004/transcribe
```

---

## Expected Server Behavior

### VAD Filter = true
- Server should apply Voice Activity Detection
- Silence segments filtered out
- Cleaner transcription output

### VAD Filter = false
- No VAD filtering applied
- All audio segments transcribed
- May include silence/noise

### enable_diarization = true
- Server attempts speaker diarization
- May log warnings if not fully implemented
- Results may include speaker labels

### enable_diarization = false
- No speaker identification
- Single speaker assumed
- Faster processing

---

## Check Server Logs

Monitor server logs for parameter values:

```bash
tail -f server_log.txt
```

Look for:
```
vad_filter=True/False
enable_diarization=True/False
```

---

## Browser Console Testing

1. Open browser DevTools (F12)
2. Go to Console tab
3. Upload file and submit
4. Check FormData being sent:

```javascript
// You should see in Network tab:
vad_filter: true
enable_diarization: false
```

---

## Rollback Instructions

If anything breaks, restore backup:

```bash
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/static

# Restore backup
cp index.html.backup-20251101-194638 index.html

# Restart server
cd /home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx
python3 server_production.py
```

---

## Success Criteria

✅ Both checkboxes display correctly
✅ VAD checked by default, Diarization unchecked
✅ Descriptions appear under each checkbox
✅ Results show correct ON/OFF status with color coding
✅ Server receives correct vad_filter and enable_diarization values
✅ All existing functionality still works
✅ Works in all three themes (Light, Dark, Unicorn)

---

## Files Modified

- `/home/ucadmin/UC-1/Unicorn-Amanuensis/whisperx/static/index.html`
- Backup: `index.html.backup-20251101-194638`

## Documentation

- `GUI_UPDATE_SUMMARY_20251101.txt` - Complete change summary
- `GUI_BEFORE_AFTER_20251101.txt` - Before/after comparison
- `QUICK_TEST_GUIDE.md` - This file

---

## Support

If issues occur:
1. Check backup exists: `ls -lh whisperx/static/*.backup*`
2. Verify server running: `curl http://localhost:9004/status`
3. Check browser console for JavaScript errors
4. Review server logs for parameter values

---

**Date**: November 1, 2025
**File**: `/home/ucadmin/UC-1/Unicorn-Amanuensis/QUICK_TEST_GUIDE.md`
**Status**: Ready for testing
