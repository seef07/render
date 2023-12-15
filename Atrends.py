import requests
from bs4 import BeautifulSoup

# URL of the website you want to scrape
url = 'https://trends.google.com/trends/explore?date=now+7-d&q=bitcoin+sell'  # Replace with the actual URL

# Send a GET request to the URL
response = requests.get(url)

# Parse the HTML content using Beautiful Soup
soup = BeautifulSoup(response.content, 'html.parser')

print(soup)
# Find the elements with class 'linechart' (assuming the class is 'linechart')
linechart_elements = soup.find_all(class_='line-chart')

# Extract the path data from the linechart elements
for element in linechart_elements:
    # Assuming the path data is within 'path' tags
    path_data = element.find('path')['d']
    print(path_data)  # This will print out the path data
