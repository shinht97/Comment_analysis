# 📝 Review Analyzer

**Review Analyzer**는 상품 리뷰 데이터를 분석하여 감정(긍정/부정)을 분류하고, 해당 리뷰에 예상되는 점수를 예측하는 자연어 처리 기반의 데스크탑 애플리케이션입니다. 두 개의 자연어 처리 모델을 사용하여 리뷰의 반응을 분석하고, 해당 리뷰에 달릴 점수를 예측합니다.

---

## 🔍 주요 기능

- **리뷰 감정 분석**: 리뷰 텍스트를 입력하면 긍정 또는 부정으로 분류합니다.
- **점수 예측**: 입력된 리뷰에 대해 예상되는 점수를 예측합니다.
- **데스크톱 어플리케이션**: 프로그램의 실행 부분이 어플리케이션으로 동작합니다.

---

## 🖥️ 사용자 인터페이스

애플리케이션은 PyQt를 사용하여 개발되었으며, 직관적인 GUI를 제공합니다.

![무반응 댓글 분석](https://github.com/shinht97/Comment_analysis/assets/71716980/b68a05dc-6226-4088-9f97-b4876a350e32)  
*무반응 댓글 분석 화면*

![긍정 댓글 분석](https://github.com/shinht97/Comment_analysis/assets/71716980/d61e23d4-fabf-435c-9066-d07f63a1f660)  
*긍정 댓글 분석 화면*

---

## 📁 프로젝트 구조

```
Comment_analysis/
├── image/               # 분석 결과 이미지
├── models/              # 학습된 모델 파일
├── source_code/         # 주요 소스 코드
├── ui/                  # PyQt UI 파일
├── validate_data/       # 검증용 데이터
├── 발표자료/              # 발표 자료
├── .gitignore
├── README.md
├── requirements.txt     # 필요한 패키지 목록
├── stopwords.csv        # 불용어 리스트
```

---

## ⚙️ 설치 및 실행 방법

1. **필수 패키지 설치**

   ```bash
   pip install -r requirements.txt
   ```

2. **애플리케이션 실행**

   ```bash
   python source_code/main.py
   ```

---

## 🛠 사용 기술

- **언어**: Python
- **라이브러리**: PyQt, scikit-learn, pandas, numpy 등
- **모델**: 자연어 처리 기반 감정 분석 및 점수 예측 모델

---

## 📄 참고 자료

- [stopwords.csv](https://github.com/shinht97/Comment_analysis/blob/main/stopwords.csv): 불용어 리스트
- [requirements.txt](https://github.com/shinht97/Comment_analysis/blob/main/requirements.txt): 필요한 패키지 목록

---

본 프로젝트에 대한 자세한 내용은 [GitHub 리포지토리](https://github.com/shinht97/Comment_analysis)를 참고하세요.

<!-- # Review_Analyzer


리뷰를 분석 하여 학습한 모델을 이용 하여 새로운 리뷰의 반응을 파악하고 그 리뷰에 달릴 점수를 예상하는 프로그램

두개의 자연어 처리 모델을 사용하여, 리뷰의 반응을 분석하는 모델과 해당하는 리뷰에 달릴 점수를 예상하는 모델을 사용하여 처리

=========================================================

### UI를 이용한 어플리케이션


![image](https://github.com/shinht97/Comment_analysis/assets/71716980/b68a05dc-6226-4088-9f97-b4876a350e32)  
<무반응 댓글 분석>


![image](https://github.com/shinht97/Comment_analysis/assets/71716980/d61e23d4-fabf-435c-9066-d07f63a1f660)  
<긍정 댓글 분석> -->

