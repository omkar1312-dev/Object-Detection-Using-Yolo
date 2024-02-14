import requests

# Set the API endpoint
api_url = "http://127.0.0.1:5000/process_image"

# Prepare the image file
image_file_path = "F:\Assignment\Infilect\project\image-10.jpg"
image_file = {'image': open(image_file_path, 'rb')}

try:
    # Send the POST request
    response = requests.post(api_url, files=image_file)

    # Check the response
    if response.status_code == 200:
        print("Success!")
        print(response.json())
    else:
        print(f"Error: {response.status_code}")
        print(response.json())

except Exception as e:
    print(f"An error occurred: {str(e)}")
