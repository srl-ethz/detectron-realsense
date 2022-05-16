import math

def get_record_counter(file):
        # Determine the record counter
        with open(file, 'r+', encoding='utf8') as f:
            # Process lines and get first line, there should only be one line
            lines = (line.strip() for line in f if line)
            x = [int(float(line.replace('\x00', ''))) for line in lines]
            ret = x[0]

            # Delete all file contents
            f.truncate(0)

            # Write back to file beginning
            f.seek(0)
            f.write(str(ret + 1))

        return ret

def truncate(number, digits) -> float:
        stepper = 10.0 ** digits
        return math.trunc(stepper * number) / stepper


RECORD_COUNTER = get_record_counter('counter')
LOG_NAME = 'test'
LOG_FILE = f'logs/{LOG_NAME}_{RECORD_COUNTER}.csv'
VIDEO_FILE = f'videos/{LOG_NAME}_{RECORD_COUNTER}.avi'
VIDEO_RAW_FILE = f'videos/{LOG_NAME}_{RECORD_COUNTER}_raw.avi'
VIDEO_DEPTH_FILE = f'videos/{LOG_NAME}_{RECORD_COUNTER}_depth.avi'
