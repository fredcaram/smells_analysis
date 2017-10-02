import urllib3

class http_helper:
    def http_get_request(self,url):
        http = urllib3.PoolManager()
        return http.request('GET', url)

    def http_get_data(self,url):
        req = self.http_get_request(url)
        if req.status == 200:
            return req.data.decode('utf-8')
        raise Exception("Invalid Request")
