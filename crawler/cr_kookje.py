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

def crawl_kookje_final_timestamp_match():
    print("=" * 60)
    print("   국제대학교 기숙사 크롤러 (목록 날짜 + 상세 내용 매칭)")
    print("   목록의 tr 태그에서 날짜(td[4]/p)를 추출하여 본문과 합칩니다.")
    print("=" * 60)

    # 1. 목록 페이지 URL 입력
    default_url = "https://www.kookje.ac.kr/dorm/index.php?pCode=failreport"
    target_list_url = input(f"1. 게시판 목록 URL을 입력하세요 (엔터치면 기본값): ").strip()
    
    if not target_list_url:
        target_list_url = default_url

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
    save_folder = input_folder if input_folder else "croldata"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'
    }

    print("-" * 60)
    print(f"[INFO] '{target_list_url}' 크롤링 시작...")
    print("-" * 60)

    total_count = 0

    # --- [Step 1] 목록 페이지 순회 ---
    for page in range(start_page, end_page + 1):
        separator = "&" if "?" in target_list_url else "?"
        current_list_url = f"{target_list_url}{separator}pg={page}"

        print(f"\n[List] {page}페이지 읽는 중... ({current_list_url})")

        try:
            resp = requests.get(current_list_url, headers=headers, verify=False)
            tree = html.fromstring(resp.content)

            # -----------------------------------------------------------
            # [목록 XPath 설정] 제공해주신 정보 적용
            # -----------------------------------------------------------
            # 1. 게시글 행(Row) 전체를 찾습니다. (tbody 안의 모든 tr)
            rows = tree.xpath('//tbody/tr')

            if not rows:
                print("   [WARNING] 게시글 목록(tr)을 찾지 못했습니다.")
                break

            items_to_crawl = []

            for row in rows:
                try:
                    # 2. 링크 추출 (현재 행(row) 안에서 idx가 있는 a태그 찾기)
                    link_nodes = row.xpath('.//a[contains(@href, "idx=")]/@href')
                    if not link_nodes: continue
                    link = link_nodes[0]
                    
                    if "download" in link: continue # 첨부파일 링크 제외

                    # 3. 날짜 추출 (현재 행(row) 안의 4번째 td 안의 p태그)
                    # 제공해주신 경로: .../tr[1]/td[4]/p
                    date_nodes = row.xpath('./td[4]/p/text()')
                    
                    if date_nodes:
                        raw_date = date_nodes[0].strip() # 예: 2024-11-28
                    else:
                        raw_date = "00000000"

                    full_url = urljoin(current_list_url, link)
                    
                    # (URL, 날짜) 짝지어서 리스트에 담기
                    items_to_crawl.append((full_url, raw_date))

                except Exception as e:
                    continue
            
            # 중복 제거 (URL 기준)
            unique_items = {url: date for url, date in items_to_crawl}
            
            print(f"   -> {len(unique_items)}개의 게시글 발견 (날짜 정보 획득 완료)")

            # --- [Step 2] 상세 페이지 접속 ---
            for view_url, date_str in unique_items.items():
                match = re.search(r'idx=(\d+)', view_url)
                post_id = match.group(1) if match else str(int(time.time()))
                
                # ★ 상세 페이지로 날짜(date_str)를 넘겨줍니다.
                process_detail_page(view_url, post_id, date_str, headers, save_folder)
                total_count += 1
                
                time.sleep(random.uniform(0.5, 1.0))

        except Exception as e:
            print(f"   [ERROR] 목록 처리 중 오류: {e}")

    print("=" * 60)
    print(f"[INFO] 전체 완료! 총 {total_count}개의 파일을 저장했습니다.")


def process_detail_page(url, post_id, list_date, headers, save_folder):
    """
    list_date: 목록 페이지에서 가져온 날짜 (예: 2025-11-28)
    """
    try:
        resp = requests.get(url, headers=headers, verify=False)
        resp.encoding = 'utf-8'
        
        if resp.status_code != 200:
            print(f"     [실패] 접속 오류 {resp.status_code}")
            return
        
        tree = html.fromstring(resp.content)

        # ------------------------------------------------------------------
        # [상세 페이지 XPath 설정] - 이전 대화 내용 반영
        # ------------------------------------------------------------------

        # 1. Title (일반적인 h3 태그 사용)
        title_tag = tree.xpath('//h3')
        if not title_tag:
            # 제목을 못 찾으면 스킵
            print(f"     [스킵] 제목 미발견")
            return
        title = title_tag[0].text_content().strip()

        # 2. Poster (상대경로: .pwriter 클래스)
        poster_xpath = '//span[contains(@class, "pwriter")]'
        poster_elements = tree.xpath(poster_xpath)
        poster = ' '.join(poster_elements[0].text_content().split()) if poster_elements else ""

        # 3. Content (상대경로: id="boardContents")
        content_xpath = '//div[@id="boardContents"]'
        content_elements = tree.xpath(content_xpath)
        
        if content_elements:
            target = content_elements[0]
            # 스크립트/스타일 제거
            for s in target.xpath('.//script | .//style'): s.drop_tree()
            # 공백 정리
            content = ' '.join(target.text_content().split())
        else:
            content = ""

        # ------------------------------------------------------------------
        # [날짜 처리] 목록에서 가져온 list_date 사용
        # ------------------------------------------------------------------
        timestamp = list_date  # 예: "2024-11-28"
        
        # YYYYMMDD 포맷으로 변환 (파일명용)
        date_numbers = re.findall(r'\d+', timestamp)
        date_prefix = "00000000"
        
        if len(date_numbers) >= 3:
            year = date_numbers[0]
            if len(year) == 2: year = "20" + year # 25 -> 2025
            date_prefix = f"{year}{int(date_numbers[1]):02d}{int(date_numbers[2]):02d}"

        # 파일명 생성: YYYYMMDD_ID.json
        filename = f"{date_prefix}_{post_id}.json"
        filepath = os.path.join(save_folder, filename)

        post_data = {
            "url": url,
            "id": post_id,
            "title": title,
            "timestamp": timestamp, # 목록에 있던 날짜
            "poster": poster,
            "content": content
        }

        with open(filepath, 'w', encoding='utf-8-sig') as f:
            json.dump(post_data, f, ensure_ascii=False, indent=4)
        
        print(f"     [저장 완료] '{title}' ({timestamp})")

    except Exception as e:
        print(f"     [ERROR] 상세 페이지 에러: {e}")

if __name__ == "__main__":
    crawl_kookje_final_timestamp_match()