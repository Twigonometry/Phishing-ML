import sys
import mailbox
import csv
from email.header import decode_header

# credit: https://stackoverflow.com/questions/63069682/decode-and-access-mbox-file-with-mbox-python-mdule

infile = "./Data/emails-phishing.mbox"
outfile = "./Data/phishing_mbox.csv"
writer = csv.writer(open(outfile, "w"), escapechar='\\')

from email.parser import BytesParser
from email.policy import default
import mailbox

mbox = mailbox.mbox(infile, factory=BytesParser(policy=default).parse)

writer.writerow(['Body'])
for _, message in enumerate(mbox):
    try:
        if message.is_multipart():
            contents = []
            for part in message.walk():
                maintype = part.get_content_maintype()
                if maintype == 'multipart' or maintype != 'text':
                    # Reject containers and non-text types
                    continue
                contents.append(part.get_content())
            content = '\n\n'.join(contents)
        else:
            content = message.get_content()

        row = [
            content
        ]
        writer.writerow(row)
    except LookupError:
        continue