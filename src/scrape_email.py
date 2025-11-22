from pydantic import BaseModel
import imaplib
import email
from email.header import decode_header
import os
from config import ENV_CONFIG


class ParsedEmail(BaseModel):
    sender: str
    header: str
    body: str

    def to_txt(self) -> str:
        return f"from: {self.sender}\nheader: {self.header}\nbody:\n{self.body}\n"


def decode_maybe(value):
    if not value:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode()
        except:
            decoded, enc = decode_header(value)[0]
            if isinstance(decoded, bytes):
                return decoded.decode(enc or "utf-8", errors="ignore")
            return decoded
    return str(value)


def extract_plain_body(msg):
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    return payload.decode(errors="ignore")
        return ""
    else:
        payload = msg.get_payload(decode=True)
        return payload.decode(errors="ignore") if payload else ""


def fetch_all_emails(email_id, password, folder="INBOX"):
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(email_id, password)
    mail.select(folder)

    status, messages = mail.search(None, "ALL")
    email_ids = messages[0].split()

    for eid in email_ids:
        _, msg_data = mail.fetch(eid, "(RFC822)")
        raw_msg = msg_data[0][1]
        msg = email.message_from_bytes(raw_msg)

        sender = decode_maybe(msg.get("From"))
        subject = decode_maybe(msg.get("Subject"))
        body = extract_plain_body(msg)

        yield ParsedEmail(sender=sender, header=subject, body=body)

    mail.logout()


def save_email_as_txt(email_obj: ParsedEmail, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(email_obj.to_txt())


def main():
    EMAIL = ENV_CONFIG.EMAIL_ADDRESS
    APP_PASSWORD = ENV_CONFIG.EMAIL_PASSWORD
    OUT_DIR = os.path.join("data", "extracted", "students")

    os.makedirs(OUT_DIR, exist_ok=True)

    for idx, email_obj in enumerate(fetch_all_emails(EMAIL, APP_PASSWORD), start=1):
        filepath = os.path.join(OUT_DIR, f"email_{idx}.txt")
        save_email_as_txt(email_obj, filepath)


if __name__ == "__main__":
    main()
