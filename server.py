import io
import http.server
import socketserver
from PIL import Image
import torch
from model import Extractor, MyModel


extractor = Extractor()
# 载入模型权重
model = MyModel.load_from_checkpoint('pretrained/loss=0.16-epoch=999-step=80000.ckpt')


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
                feature = extractor(extractor.transform(img).unsqueeze(0)).detach().cpu().reshape(-1)
                out = torch.argmax(model(feature))
            resp = str.encode(str(out.item()))
        except:
            resp = str.encode('-1')

        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)


PORT = 3000
with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
    print("Serving at port", PORT)
    httpd.serve_forever()
