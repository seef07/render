from tweety import Twitter
username = "Saif99632023"
password = "d6DfgRCX@s5sfdPOI"
app = Twitter("session")
app.sign_in(username, password)

tweets = app.search('bitcoin', pages =  2)
for tweet in tweets:
    print("Time:", tweet.created_on)
    print("Tweet:", tweet.text)  # Assuming there is a 'text' attribute for the tweet content
    print("\n")