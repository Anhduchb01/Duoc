import re
from datetime import datetime
import dateutil.parser

def convert_to_custom_format(datestring):
    # Define a list of formats to check
    formats_to_check = [
        "%d-%m-%Y - %H:%M %p",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d %I:%M %p",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y %I:%M %p",
    ]

    cleaned_datestring = re.sub(r'[^0-9/:\-]', ' ', datestring)
    cleaned_datestring = re.sub(r'\s+', ' ', cleaned_datestring).strip()

    try:
        yourdate = dateutil.parser.parse(cleaned_datestring)
        formatted_date = yourdate.strftime("%Y/%m/%d")
        return formatted_date
    except ValueError:
        for format_string in formats_to_check:
            try:
                datetime_object = datetime.strptime(datestring, format_string)
                formatted_date = datetime_object.strftime('%Y/%m/%d')
                return formatted_date
            except ValueError:
                pass
        
        return None  # If no valid format is found