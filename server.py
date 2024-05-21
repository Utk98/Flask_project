from app import app
# from pyngrok import ngrok

if __name__ == '__main__':
    # public_url = ngrok.connect(5000).public_url
    # print(" * Running on", public_url)
    app.run(host='0.0.0.0', port=5000)
