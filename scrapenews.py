import requests
from bs4 import BeautifulSoup

# URL of the website
url = "https://phoenixnews.io/"

# Send an HTTP request to the URL
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    print("suc6")
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    print(soup)
    # Extract information based on HTML structure
    # Replace this part with your own logic to find and extract the data you need

    # Example: Extracting text from all paragraphs
    paragraphs = soup.find_all('p')
    for paragraph in paragraphs:
        print(paragraph.text)

    # Example: Extracting the value of a specific class
    specific_class = soup.find('div', class_='ycard-crypto-color card')
    if specific_class:
        print(specific_class.text)
else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")
