import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex

email_1 = "*This message was transferred with a trial version of CommuniGate(tm) Pro*\\nFree Debt Consolidation InformationFree 1 Minute Debt Consolidation Quote\\n* Quickly and easily reduce Your Monthly Debt Payments Up To 60% \\nWe are a 501c Non-Profit Organization that has helped 1000's consolidate their debts into one easy affordable monthly payment. For a Free - No Obligation quote to see how much money we can save you, please read on.Become Debt Free...Get Your Life Back On Track!All credit accepted and home \\nownership is NOT required.\\n Not Another Loan To Dig You Deeper In To Debt!Â• 100% Confidential - No Obligation - Free Quote Â• Free Debt Consolidation Quote\\nIf you have $4000 or more in debt, a trained professional will negotiate with your creditors to:Lower your monthly debt payments up to 60%End creditor harassmentSave thousands of dollars in interest and late chargesStart improving your credit ratingComplete\\nOur QuickForm and Submit it for Your Free Analysis.\\nName Street Address \\nCity State / Zip  Alabama\\n Alaska\\n Arizona\\n Arkansas\\n California\\n Colorado\\n Connecticut\\n Delaware\\n Dist of Columbia\\n Florida\\n Georgia\\n Hawaii\\n Idaho\\n Illinois\\n Indiana\\n Iowa\\n Kansas\\n Kentucky\\n Louisiana\\n Maine\\n Maryland\\n Massachusetts\\n Michigan\\n Minnesota\\n Mississippi\\n Missouri\\n Montana\\n Nebraska\\n Nevada\\n New Hampshire\\n New Jersey\\n New Mexico\\n New York\\n North Carolina\\n North Dakota\\n Ohio\\n Oklahoma\\n Oregon\\n Pennsylvania\\n Rhode Island\\n South Carolina\\n South Dakota\\n Tennessee\\n Texas\\n Utah\\n Vermont\\n Virginia\\n Washington\\n West Virginia\\n Wisconsin\\n Wyoming \\nHome Phone (with area code) Work Phone (with area code) Best\\nTime To Contact Morning at Home\\nMorning at Work\\nAfternoon at Home\\nAfternoon at Work\\nEvening at Home\\nEvening at Work \\nWeekend\\nEmail\\naddress \\nTotal\\nDebt $4000 - $4999\\n$5000 - $7500\\n$7,501 - $10,000\\n$10,001 - $12,500\\n$12,501 - $15,000\\n$15,001 - $17,500\\n$17,501 - $20,000\\n$20,001 - $22,500\\n$22,501 - $25,000\\n$25,001 - $27,500\\n$27,501 - $30,000\\n$30,001 - $35,000\\n$35,001 - $40,000\\n$45,001 - $50,000\\n$50,000+Please click the submit button\\njust once - process will take 30-60 seconds.\\nOr, please reply to the email with the following for your Free Analysis\\nName:__________________________\\nAddress:________________________\\nCity:___________________________\\nState:_____________ Zip:_________\\nHome Phone:(___) ___-____    \\nWork Phone:(___) ___-____    \\nBest Time:_______________________\\nEmail:___________________________\\nTotal Debt:______________________\\nNot Interested?  Please send and email to jayshilling4792@excite.commamrfitxhidjpfxo\\nhttp://xent.com/mailman/listinfo/fork\\n"

email_2 = "Shorter email more indicative of a non-phishing email"

sid_obj = SentimentIntensityAnalyzer()

def label_vader(email):
    return sid_obj.polarity_scores(email)

def label_nrc(email):
    return NRCLex(email.lower().replace("\\n", " "))

t1 = time.time()

label_vader(email_1)

t2 = time.time()

print(f"Label vader email 1 {t2 - t1} seconds")

t3  = time.time()

label_vader(email_2)

t4 = time.time()

print(f"Label vader email 2 {t4 - t3} seconds")

t1 = time.time()

label_nrc(email_1)

t2 = time.time()

print(f"Label nrc email 1 {t2 - t1} seconds")

t3  = time.time()

label_nrc(email_2)

t4 = time.time()

print(f"Label nrc email 2 {t4 - t3} seconds")