# to run 
# scrapy crawl tmdb_spider -o movies.csv -a subdir=671-harry-potter-and-the-philosopher-s-stone

import scrapy

class TmdbSpider(scrapy.Spider):
    name = 'tmdb_spider'

    def __init__(self, subdir="", *args, **kwargs):
        self.start_urls = [f"https://www.themoviedb.org/movie/{subdir}/"]

    def parse(self, response):
        """starting from a movie page on themoviedb.org, navigates to the Full Cast & Crew page

        i.e. starting from f"https://www.themoviedb.org/movie/{subdir}"
               go to f"https://www.themoviedb.org/movie/{subdir}/cast"

        Args:
            self (Spider): the instance of the spider calling this method
            response (scrapy.http.Response): the response object containing the movie page data, 
                                            from which the cast page URL will be derived and followed.
        """

        # to get url for Full Cast & Crew page
        cast_url = self.start_urls[0] + "cast"

        # navigates to Full Cast & Crew page
        yield scrapy.http.Request(url = cast_url, 
                                    callback = self.parse_full_credits)

        
    def parse_full_credits(self, response):
        """assuming we start on Full Cast & Crew Page, yields a scrapy.Request for each actor listed on 
            this page

        Args:
            self (Spider): the instance of the spider calling this method
            response (scrapy.http.Response): the response object containing the movie page data, 
                                            from which the cast page URL will be derived and followed.
        """
        # to get a list of all cast names (excluding crew)
        actors =  response.css("ol[class='people credits '] div[class=info] a::attr(href)").getall()

        # yield a scrapy.Request for each actor
        for actor in actors:
            actor_url = "https://www.themoviedb.org" + actor
            yield scrapy.http.Request(url = actor_url,
                                        callback = self.parse_actor_page)

        
    def parse_actor_page(self, response):
        """assuming we start on page of an actor, scrapes page to get data regarding each movie/TV show that
            actor has worked in an "Acting" role
        
        Args:
            self (Spider): the instance of the spider calling this method
            response (scrapy.http.Response): the response object containing the movie page data, 
                                            from which the cast page URL will be derived and followed.

        Returns:
            a dictionary containing two key value pairs of the form
                    {"actor": actor_name, "movie_or_TV_name: movie_or_TV_name"}
        """

        # extracting actor name from page
        actor_name = response.css("h2[class=title] a::text").get()  

        # extracting list of all (unique) shows/movies actor has been in
        movie_or_TV_name = list(set(response.css("a[class=tooltip] bdi::text").getall()))
        
        # returning desired dictionary
        yield {"actor": actor_name, 
                "movie_or_TV_name": movie_or_TV_name}







