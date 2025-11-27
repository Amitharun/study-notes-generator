const fileInput = document.getElementById('fileInput');
const dropzone = document.getElementById('dropzone');
const fileInfo = document.getElementById('fileInfo');
const textInput = document.getElementById('textInput');
const summarizeBtn = document.getElementById('summarizeBtn');
const clearBtn = document.getElementById('clearBtn');
const progressSection = document.getElementById('progressSection');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const result = document.getElementById('result');
const bulletsEl = document.getElementById('bullets');
const downloadBtn = document.getElementById('downloadBtn');
const copyBtn = document.getElementById('copyBtn');
let currentBullets = [];

// file UI
dropzone.addEventListener('click', ()=> fileInput.click());
fileInput.addEventListener('change', ()=>{
  if (fileInput.files.length) fileInfo.textContent = fileInput.files[0].name;
  else fileInfo.textContent = 'No file chosen.';
});
['dragenter','dragover'].forEach(e=> dropzone.addEventListener(e, ev=>{ ev.preventDefault(); dropzone.classList.add('drag') }));
['dragleave','drop'].forEach(e=> dropzone.addEventListener(e, ev=>{ ev.preventDefault(); dropzone.classList.remove('drag') }));
dropzone.addEventListener('drop', ev=>{ ev.preventDefault(); if (ev.dataTransfer.files.length){ fileInput.files = ev.dataTransfer.files; fileInfo.textContent = ev.dataTransfer.files[0].name } });

function setProgress(p, text){ progressSection.classList.remove('hide'); progressFill.style.width = p + '%'; progressText.textContent = text; }
function resetProgress(){ progressFill.style.width = '0%'; progressText.textContent = ''; progressSection.classList.add('hide'); }

function simulateProgressDuringFetch(controller){
  // Simple progressive animation until fetch completes.
  let val = 5;
  setProgress(val, 'Starting...');
  const id = setInterval(()=>{
    if (val < 85) val += Math.random()*6;
    else val += Math.random()*1.5;
    if (val > 98) val = 98;
    setProgress(Math.floor(val), 'Summarizing...');
  }, 400);
  return ()=>{ clearInterval(id); setProgress(100,'Completed'); setTimeout(()=>resetProgress(), 700); }
}

async function summarize(){
  bulletsEl.innerHTML = ''; result.classList.add('hide');
  const form = new FormData();
  if (textInput.value.trim()) form.append('text', textInput.value.trim());
  else if (fileInput.files[0]) form.append('file', fileInput.files[0]);
  else { alert('Please provide text or a file first.'); return; }

  summarizeBtn.disabled = true; clearBtn.disabled = true; summarizeBtn.textContent = 'Summarizing...';
  const stopProgress = simulateProgressDuringFetch();
  try{
    const res = await fetch('/api/process', { method:'POST', body: form });
    stopProgress();
    summarizeBtn.textContent = 'Summarize';
    summarizeBtn.disabled = false; clearBtn.disabled = false;
    if (!res.ok){ const err = await res.json(); alert(err.error || 'Error'); return; }
    const data = await res.json();
    currentBullets = data.summary_bullets || [];
    if (currentBullets.length === 0){ alert('No summary produced.'); return; }
    currentBullets.forEach(b=>{ const li = document.createElement('li'); li.textContent = b; bulletsEl.appendChild(li); });
    result.classList.remove('hide');
  }catch(e){
    stopProgress();
    summarizeBtn.disabled = false; clearBtn.disabled = false; summarizeBtn.textContent = 'Summarize';
    alert('Network or server error: ' + e.message);
  }
}

summarizeBtn.addEventListener('click', summarize);
clearBtn.addEventListener('click', ()=>{ textInput.value=''; fileInput.value=''; fileInfo.textContent='No file chosen.'; bulletsEl.innerHTML=''; result.classList.add('hide'); });

downloadBtn.addEventListener('click', ()=>{
  const txt = currentBullets.map(b=>'• '+b).join('\n');
  const blob = new Blob([txt], {type:'text/plain'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = 'summary.txt'; a.click(); URL.revokeObjectURL(url);
});

copyBtn.addEventListener('click', ()=>{
  const txt = currentBullets.map(b=>'• '+b).join('\n');
  navigator.clipboard.writeText(txt).then(()=>alert('Copied to clipboard'));
});
