import requests
from lxml import html
import json
import urllib3
import os
import re
import time
import random
from urllib.parse import urljoin

# SSL 경고 무시
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def crawl_board_flexible_content():
    print("=" * 60)
    print("   강원대 게시판 크롤러 (Content 위치 자동 대응 버전)")
    print("=" * 60)

    # 1. 목록 페이지 URL 입력
    target_list_url = input("1. 게시판 목록 URL을 입력하세요: ").strip()
    
    if not target_list_url:
        print("[ERROR] URL을 입력해야 합니다.")
        return

    if not target_list_url.startswith(("http://", "https://")):
        target_list_url = "https://" + target_list_url

    # 2. 페이지 범위 설정
    try:
        start_page = int(input("2. 시작 페이지 (예: 1): "))
        end_page = int(input("3. 종료 페이지 (예: 3): "))
    except ValueError:
        print("[ERROR] 숫자를 입력하세요.")
        return

    # 3. 저장 폴더 입력
    input_folder = input("4. 저장할 폴더명을 입력하세요 (엔터치면 'croldata' 사용): ").strip()
    
    if input_folder:
        save_folder = input_folder
    else:
        save_folder = "croldata"

    # 폴더 생성
    if not os.path.exists(save_folder):
        try:
            os.makedirs(save_folder)
        except OSError as e:
            print(f"[ERROR] 폴더 생성 실패: {e}")
            return

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'
    }

    print("-" * 60)
    print(f"[INFO] '{target_list_url}' 크롤링 시작...")
    print(f"[INFO] 저장 위치: ./{save_folder}/")
    print("-" * 60)

    total_count = 0

    # --- [Step 1] 목록 페이지 순회 ---
    for page in range(start_page, end_page + 1):
        separator = "&" if "?" in target_list_url else "?"
        current_list_url = f"{target_list_url}{separator}pageIndex={page}"

        print(f"\n[List] {page}페이지 읽는 중... ({current_list_url})")

        try:
            resp = requests.get(current_list_url, headers=headers, verify=False)
            tree = html.fromstring(resp.content)

            # --- [Step 2] 게시글 링크 추출 ---
            link_xpath = '//a[contains(@href, "nttNo=")]/@href'
            raw_links = tree.xpath(link_xpath)

            if not raw_links:
                print("   [WARNING] 게시글 링크를 찾지 못했습니다.")
                break

            full_links = []
            for link in raw_links:
                if "downloadBbsFile" in link or "preview" in link:
                    continue
                
                full_url = urljoin(current_list_url, link)
                full_links.append(full_url)
            
            full_links = list(set(full_links))
            print(f"   -> {len(full_links)}개의 게시글(파일 제외) 발견")

            # --- [Step 3] 상세 페이지 접속 및 저장 ---
            for view_url in full_links:
                match = re.search(r'nttNo=(\d+)', view_url)
                ntt_id = match.group(1) if match else str(int(time.time()))
                
                process_detail_page(view_url, ntt_id, headers, save_folder)
                total_count += 1
                
                time.sleep(random.uniform(0.5, 1.0))

        except Exception as e:
            print(f"   [ERROR] 목록 처리 중 오류: {e}")

    print("=" * 60)
    print(f"[INFO] 전체 완료! 총 {total_count}개의 파일을 '{save_folder}' 폴더에 저장했습니다.")


def process_detail_page(url, ntt_id, headers, save_folder):
    """상세 페이지 내용을 긁어서 JSON으로 저장"""
    try:
        resp = requests.get(url, headers=headers, verify=False)
        resp.encoding = 'utf-8'
        
        if resp.status_code != 200:
            return
        
        tree = html.fromstring(resp.content)

        # ------------------------------------------------------------------
        # [XPath 설정]
        # ------------------------------------------------------------------

        # 1. Title
        title_xpath = '/html/body/div[1]/div[2]/div[2]/main/article/div/div/table/tbody/tr[1]/td'
        title_tag = tree.xpath(title_xpath)
        if not title_tag:
            return 
        title = title_tag[0].text_content().strip()

        # 2. Poster
        poster_xpath = '/html/body/div[1]/div[2]/div[2]/main/article/div/div/table/tbody/tr[4]/td/text()'
        poster_results = tree.xpath(poster_xpath)
        poster = ' '.join(' '.join(poster_results).split())

        # 3. Content (수정됨: div[2]를 빼고 td 전체를 잡음)
        # 기존: .../tr[2]/td/div[2]
        # 변경: .../tr[2]/td  (td 안에 있는 모든 텍스트를 대상으로 함)
        content_xpath = '/html/body/div[1]/div[2]/div[2]/main/article/div/div/table/tbody/tr[2]/td'
        content_elements = tree.xpath(content_xpath)
        
        if content_elements:
            target_element = content_elements[0]
            
            # td 안에 섞여있는 스크립트/스타일 제거
            for script in target_element.xpath('.//script | .//style'):
                script.drop_tree()
            
            # td 안의 모든 텍스트 추출 (div[1]이든 div[2]든 다 가져옴)
            raw_content = target_element.text_content()
            content = ' '.join(raw_content.split())
        else:
            content = ""

        # 4. Timestamp
        timestamp_xpath = '/html/body/div[1]/div[2]/div[2]/main/article/div/div/div[1]/div/span[2]/strong'
        timestamp_elements = tree.xpath(timestamp_xpath)
        
        if timestamp_elements:
            timestamp = timestamp_elements[0].text_content().strip()
            date_numbers = re.findall(r'\d+', timestamp)
            if len(date_numbers) >= 3:
                date_prefix = f"{date_numbers[0]}{int(date_numbers[1]):02d}{int(date_numbers[2]):02d}"
            else:
                date_prefix = "00000000"
        else:
            timestamp = "날짜정보없음"
            date_prefix = "00000000"

        # ------------------------------------------------------------------

        filename = f"{date_prefix}_{ntt_id}.json"
        filepath = os.path.join(save_folder, filename)

        post_data = {
            "url": url,
            "id": ntt_id,
            "title": title,
            "timestamp": timestamp,
            "poster": poster,
            "content": content
        }

        with open(filepath, 'w', encoding='utf-8-sig') as f:
            json.dump(post_data, f, ensure_ascii=False, indent=4)
        
        print(f"     [저장 완료] '{title}' {date_prefix}")

    except Exception as e:
        print(f"     [ERROR] 상세 페이지 에러 ({ntt_id}): {e}")

if __name__ == "__main__":
    crawl_board_flexible_content()