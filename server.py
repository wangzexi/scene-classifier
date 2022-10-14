import io
import http.server
import socketserver
from PIL import Image
import torch
from model import Extractor, MyModel
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

extractor = Extractor().to(device)
# 载入模型权重
model = MyModel.load_from_checkpoint('pretrained/loss=0.16-epoch=999-step=80000.ckpt').to(device)


class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):

    def do_POST(self):
        content_len = int(self.headers.get('Content-Length'))
        body = self.rfile.read(content_len)

        # with open('upload.jpg', 'wb') as f:
        #     f.write(body)

        # 分类推理
        try:
            img = Image.open(io.BytesIO(body)).convert('RGB')
            with torch.no_grad():
                t = time.time()
                feature = extractor(extractor.transform(img).to(device).unsqueeze(0)).reshape(-1)
                out = torch.argmax(model(feature))

            res = str(out.item())
            print(f'🌈 {res} ({time.time() - t:.6f}s on {device})')
        except:
            res = '-1'
            print(f'🌧️ {res}')

        data = str.encode(res)
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


PORT = 3000
with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
    print("Serving at port", PORT)
    httpd.serve_forever()
