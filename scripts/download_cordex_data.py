from concurrent.futures import ThreadPoolExecutor

from clima_data.cordex import create_tasks, download_cordex

DATADIR = "~/data/cordex"


def download_all_data():
    variables = [
        "pr",
        "tas",
        "sfcWind",
        "tasmax",
        "tasmin",
        "rsds",
        "hurs",
    ]
    tasks = create_tasks(variables)

    download_process = lambda task: download_cordex(**task, verbose=True, data_dir=DATADIR)

    with ThreadPoolExecutor(max_workers=3) as executor:  # Respect CDS limits
        results = executor.map(download_process, tasks)
        # Output results
        for result in results:
            print(result)
    return tasks


if __name__ == "__main__":
    tasks = download_all_data()
    print(f"DONE {len(tasks)} API calls to CDS and downloading CORDEX data to {DATADIR}")
