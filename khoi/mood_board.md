# Sub-Question: User Engagement Patterns

## Question 1: What percentage of users have multiple conversations?
- WildChat: Use combination of `hashed_ip` and `header` to estimate number of repeat users -> find out that "hashed_ip" is alone strong enough to uniquely identify a user
    - Check whether the avg turns per user is different accross different gpt models
    - 

## Question 2: What is the distribution of conversations per user?
- WildChat: For IP addresses with multiple conversations, plot histogram of conversation counts

## Question 3: Does frequency of user engagement correlate with conversation length?
- WildChat: Compare avg turns per sample for one-time vs repeat users

## Question 4: What is the most common time gap between a user's conversations?
- WildChat: For repeat users, find median time between their conversation timestamps
- Lymsys: Use assistant turn timestamps to characterize typical response times

## Question 5: What are the peak conversation times for different regions?
- WildChat: Segment conversations by country/state and find top times for each
- Lymsys: Analyze timestamp distribution by language to infer regional patterns

## Question 6: How does user retention vary by country?
- WildChat: Calculate percentage of repeat users for top countries

## Question 7: Do engagement patterns differ between desktop and mobile users?
- WildChat: Use header info to classify desktop vs mobile, compare metrics like turns per conversation and time between conversations

## Question 8: How do engagement metrics evolve over a user's lifetime?
- WildChat: For users with many conversations, plot engagement metrics over time to spot trends

## Question 9: Do conversations containing toxic/unsafe content tend to be shorter or longer?
- WildChat + Lymsys: Compare average turn lengths for conversations with openai_moderation=True vs False


## Question 10: How does the presence of PII (redacted=True) correlate with conversation topics/content?
- Wildchat: Lymsys: Analyze message text (topics) and metadata (language, model, etc) for conversations with redacted=True

## Question 11 (maybe on Anlin part): Are there temporal patterns in when toxic/unsafe conversations occur?
- WildChat: Use timestamps to see if toxic conversations cluster around certain times of day/days of week
- Lymsys: Similarly analyze distribution of unsafe conversations over time

## Question 12: Do geographic regions differ in their proportion of toxic/PII conversations?
- WildChat: Calculate toxic%/redacted% metrics segmented by state/country
- Lymsys: Use language as a proxy for region to compare across groups

## Question 13: For users with multiple conversations, how consistent are their toxicity/PII patterns?
- WildChat: Identify "repeat users" based on hashed_ip/header. Calculate % of their conversations flagged as toxic/redacted - are некоторые users consistently safer?

## Question 14: Do conversation toxicity levels change over the course of a user's engagement lifetime?
- Both: For users with many conversations, track toxicity metrics over time to see evolving patterns

## Question 15: What types of unsafe content most commonly trigger toxicity flags?
- Both: Analyze message text and metadata from conversations with openai_moderation=True
- Both: Cluster/categorize the toxic messages to identify prevalent topics/categories