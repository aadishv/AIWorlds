

@app.route('/')
def index():
    """Optional: A simple index page with a link or embedded viewer."""
    # You could embed the stream directly here using an <img> tag
    return """
    <html>
        <head><title>Webcam Stream</title></head>
        <body>
            <h1>Live Webcam Feed</h1>
            <img src="/video_feed" width="640" height="480">
            <p><a href="/video_feed">Direct Stream Link</a></p>
        </body>
    </html>
    """


if __name__ == '__main__':
    # Equivalent of the 'start' console log
    print("WebCam server starting...")
    print("Access stream at http://<your-ip>:5000/ or http://<your-ip>:5000/video_feed")

    # Run the Flask app.
    # host='0.0.0.0' makes it accessible from other devices on your network.
    # Use debug=False in production
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

    # Release the camera when the server stops (though app.run blocks)
    # This might only run if debug=True and an error occurs, or on graceful shutdown signals.
    # Proper cleanup often requires signal handling.
    print("Releasing camera...")
    camera.release()
    cv2.destroyAllWindows()  # Might not be necessary in a server context
