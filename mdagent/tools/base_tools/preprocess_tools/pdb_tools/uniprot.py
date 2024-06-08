import time
import json
import requests
from requests.adapters import HTTPAdapter, Retry


class UniProt_Converter:
    API_URL = "https://rest.uniprot.org"
    POLLING_INTERVAL = 3

    RETRIES = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
    SESSION = requests.Session()
    SESSION.mount("https://", HTTPAdapter(max_retries=RETRIES))

    def __init__(self):
        pass

    def check_response(self, response):
        try:
            response.raise_for_status()
        except requests.HTTPError:
            print(response.json())
            raise

    def submit_id_mapping(self, uniprot_id, to_db, from_db):
        request = self.SESSION.post(
            f"{self.API_URL}/idmapping/run",
            data={"from": from_db, "to": to_db, "ids": uniprot_id}
        )
        self.check_response(request)
        return request.json()["jobId"]

    def get_id_mapping_results_ready(self, job_id):
        while True:
            request = self.SESSION.get(f"{self.API_URL}/idmapping/status/{job_id}")
            self.check_response(request)
            j = request.json()
            if "jobStatus" in j and j["jobStatus"] == "RUNNING":
                print(f"Retrying in {self.POLLING_INTERVAL}s")
                time.sleep(self.POLLING_INTERVAL)
            else:
                return j

    def get_id_mapping_results_link(self, job_id):
        request = self.SESSION.get(f"{self.API_URL}/idmapping/details/{job_id}")
        self.check_response(request)
        return request.json()["redirectURL"]

    def decode_results(self, response):
        content = response.content
        file_format = response.headers.get('Content-Type', '')
        if 'application/json' in file_format:
            return json.loads(content)
        elif 'text/plain' in file_format:
            return content.decode('utf-8')
        else:
            return content

    def get_id_mapping_results(self, url):
        request = self.SESSION.get(url)
        self.check_response(request)
        return self.decode_results(request)

    def convert(self, uniprot_id, to_db, from_db="UniProtKB_AC-ID"):
        try:
            job_id = self.submit_id_mapping(uniprot_id=uniprot_id, to_db=to_db, from_db=from_db)
        except Exception:
            return f"Error - one of the input parameters is incorrect"
        result_status = self.get_id_mapping_results_ready(job_id)
        if result_status.get("results"):
            link = self.get_id_mapping_results_link(job_id)
            results = self.get_id_mapping_results(link)
            if not results or not results['results']:
                return "No results found."
            else:
                results = results["results"]
                new_ids = [result["to"] for result in results]
                return f"Associated {to_db} ids: {', '.join(new_ids)}"
        elif result_status.get("jobStatus"):
            print(f"Job ended with status: {result_status['jobStatus']}")
        else:
            print("No results found.")

# Example usage
# # Example usage
uniprot_id = "P12345"
to_db = "KEGG"
converter = UniProt_Converter()
print (converter.convert(uniprot_id, to_db))

to_db = "nonsese"
print (converter.convert(uniprot_id, to_db))

to_db = "PDB"
print (converter.convert("p68871", to_db))