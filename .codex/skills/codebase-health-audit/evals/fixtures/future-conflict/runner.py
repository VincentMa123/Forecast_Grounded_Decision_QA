from config import OUTPUT_ROOT, RETRY_COUNT


def output_path(job_id):
    return f"{OUTPUT_ROOT}/{job_id}.json"


def retry_count():
    return 3
