import os
import re
import json
import requests
from bs4 import BeautifulSoup
from retry import retry
from tqdm import tqdm
from time import sleep


def sanitize_filename(filename):
    illegal_chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|', ' ', '\t', '\n', '\r']
    filename = ''.join(c for c in filename if c not in illegal_chars)
    return filename[:255]


class DiscussionItem:
    def __init__(self, author=None, content=None) -> None:
        self.author = author
        self.content = content

    def to_dict(self):
        return {
            'author': self.author,
            'content': self.content
        }


class IssueItem:
    def __init__(self, title=None, discussions=None, status=None, url=None) -> None:
        self.title = title
        self.discussions = discussions
        self.status = status
        self.source = url

    def to_dict(self):
        return {
            'title': self.title,
            'discussions': [discussion.to_dict() for discussion in self.discussions] if self.discussions else None,
            'status': self.status,
            'source': self.source
        }


class Crawler:
    def __init__(self, project_name: str, sleep_time: int=0.5, output_dir=None, debug=False):
        self.project_name = project_name
        self.sleep_time = sleep_time
        self.debug = debug
        self.output_dir = output_dir if output_dir is not None else "./issues"
        self.domain = "https://github.com"
        self.project_url = "https://github.com/{}/issues?page={}&q="

    def get_web_source(self, url):
        sleep(self.sleep_time)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return BeautifulSoup(response.content, 'html.parser')
        else:
            raise OSError("Fetching {} project web source failed,  \
                           status code is {}!".format(self.project_name, response.status_code))

    @retry(OSError, delay=1, backoff=1, max_delay=2)
    def get_issue_pages_num(self):
        if self.debug:
            return 1
        url = self.project_url.format(self.project_name, 1)
        soup = self.get_web_source(url)
        target_node = soup.find('em', class_='current')
        return int(target_node.attrs["data-total-pages"])

    def get_issue_links(self, soup):
        cur_page_issue_links = []
        issues_elements = soup.find_all(attrs={'aria-label': 'Issues', "role": "group"})
        issue_group = issues_elements[0]
        issue_list_container = issue_group.find_all('div', id=re.compile('^issue_'))
        for links in issue_list_container:
            link = links.find_all('a', attrs={"aria-label": re.compile(r'^Link to Issue')})[0]
            url = link.get("href")
            if url:
                cur_page_issue_links.append("{}{}".format(self.domain, url))
        return cur_page_issue_links
    
    def get_all_issue_links(self, page_nums):
        issue_links = []
        for i in tqdm(list(range(1, page_nums + 1)), desc="retrieving issue links"):
            url = self.project_url.format(self.project_name, i)
            try:
                soup = self.get_web_source(url)
                cur_page_issue_links = self.get_issue_links(soup)
                issue_links.extend(cur_page_issue_links)
            except Exception as e:
                print("failed to resolve page url {}, reason is {}, ignored ...".format(url, e))
        return issue_links
    
    def extract_edit_content(self, element):
        question_lines = []
        # 拼接提问
        for ele in element.find_all():
            if (ele.name == 'p' and 'mb-1' not in ele.get('class', [])) or   \
                (ele.name == 'code' and 'notranslate' in ele.get('class', [])):
                text = str(ele.text).strip()
                question_lines.append(text)
        return "\n".join(question_lines)
    
    def get_issue_verbose_info(self, url):
        issueItem = IssueItem()
        issueItem.source = url

        try:
            soup = self.get_web_source(url)
            discuss_header = soup.find(id="partial-discussion-header")
            bdi_tags = discuss_header.find_all('bdi', class_='js-issue-title markdown-title')

            title_elements = bdi_tags[0]
            issueItem.title = str(title_elements.text).strip()

            status_span = discuss_header.find_all('span', title=re.compile('^Status:'))[0]
            issueItem.status = str(status_span.text).strip()

            discussion_items = []

            discussion_bucket = soup.find(id="discussion_bucket")
            discussion_layout_outer = discussion_bucket.find_all(
                'div', recursive=False)[0]
            discussion_layout_main = discussion_layout_outer.find_all(
                class_="Layout-main", recursive=False)[0]
            
            discussion_layout_container = discussion_layout_main.find_all(
                'div', class_="js-quote-selection-container", recursive=False)[0]
            
            discussion_list = discussion_layout_container.find_all(
                'div', class_=re.compile('^js-discussion'), recursive=False)[0]
            
            timeline_comment_group = discussion_list.find_all(
                'div', class_=lambda x: 'timeline-comment-group' in x if x is not None else False
            )

            for time_line_item in timeline_comment_group:
                discussion_item = DiscussionItem()

                question = time_line_item.find_all(
                    'div', class_="edit-comment-hide"
                )
                if not question:
                    continue
                task_lists = question[0].find_all('task-lists')
                if not task_lists:
                    continue
                question_lines = self.extract_edit_content(task_lists[0])
                discussion_item.content = question_lines
                
                author = time_line_item.find_all(
                    'a', class_=re.compile('^author')
                )
                discussion_item.author = str(author[0].text).strip()
                discussion_items.append(discussion_item)

            issueItem.discussions = discussion_items
        except Exception as e:
            print("failed to resolve page url {}, reason is {}, ignored ...".format(url, e))
        return issueItem

    
    def get_issue_lists(self, serialization=True):
        # 先获取页数
        page_nums = self.get_issue_pages_num()
        # 遍历页数，把每个具体的issue链接保存下来
        issue_links = self.get_all_issue_links(page_nums)
        # 具体页面具体爬取
        issue_verbose_info = []

        file = None
        if serialization:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            filename = sanitize_filename(self.project_name) + ".jsonl"
            file = open(os.path.join(self.output_dir, filename), "a", encoding="utf-8")
        for issue_link in tqdm(issue_links, desc="retrieving issue info"):
            issueItem = self.get_issue_verbose_info(issue_link)
            issue_verbose_info.append(issueItem)
            if file:
                json.dump(issueItem.to_dict(), file)
                file.write("\n")
        if file:
            file.close()
        
        return issue_verbose_info


if __name__ == "__main__":
    project_name = "vim/vim"
    crawler = Crawler(project_name, sleep_time=0.7, debug=False)
    issueItems = crawler.get_issue_lists()
