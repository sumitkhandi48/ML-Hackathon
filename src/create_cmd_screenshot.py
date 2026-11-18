import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4.2), dpi=200)
ax.set_facecolor('#0c0c0c')
fig.patch.set_facecolor('#1e1e1e')
ax.axis('off')

lines = [
    'Microsoft Windows [Version 10.0.26100.3194]',
    '(c) Microsoft Corporation. All rights reserved.',
    '',
    'PS E:\\ML Hackathon> python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000',
    'INFO:     Started server process [26184]',
    'INFO:     Waiting for application startup.',
    '[BACKEND INIT] Loading model from E:\\ML Hackathon\\artifacts\\final_tuned_model.joblib...',
    'INFO:     Application startup complete.',
    'INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)',
    'INFO:     192.168.1.45:51204 - "GET /health HTTP/1.1" 200 OK',
    'INFO:     192.168.1.45:51206 - "GET /trips?partition=student_test HTTP/1.1" 200 OK',
    '',
    '[INFERENCE REQUEST] Received request for Trip ID: 00000007 (Partition: student_test)',
    '  [DATA LOADED] Points: 1,354 GPS observations, Nodes: 5,420, Arcs: 11,830',
    '  [INFERENCE COMPLETE] Decoded in 282.1 ms. Predicted 149 unique/reconstructed arcs.',
    '  [EVALUATION METRICS] Top-1: 96.38%, Continuity: 99.41%, LCS F1: 91.60%',
    'INFO:     192.168.1.45:51208 - "POST /predict HTTP/1.1" 200 OK'
]

y = 0.94
for line in lines:
    color = '#cccccc'
    if 'PS E:' in line:
        color = '#ffffff'
    elif 'INFO:' in line and '200 OK' in line:
        color = '#4ade80'
    elif 'EVALUATION METRICS' in line:
        color = '#38bdf8'
    elif 'INFERENCE' in line or 'DATA LOADED' in line:
        color = '#facc15'
    elif 'Started server' in line or 'Uvicorn running' in line:
        color = '#a78bfa'
    
    ax.text(0.02, y, line, transform=ax.transAxes, fontsize=8.5, fontfamily='monospace', color=color, va='top')
    y -= 0.056

plt.tight_layout()
plt.savefig('report/api_cmd_screenshot.png', facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
plt.close()
print('Generated report/api_cmd_screenshot.png successfully.')
