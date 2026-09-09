# 공개 범위와 변경 내역

2026-09-10 개인 과제의 source, config, 평가 도구, 결과와 그림을 복구했다. 팀 서비스 전체의 기여와 혼동하지 않는다. 과거 개발 이력을 새로 만들지 않고 기존 GitHub repository를 개선한다.

- `historical/git-20251214/`: 과거 commit `e86159a177f0a8b56c76c9d2dc72634184359557`의 FT 폴더. 민감한 raw pairwise CSV를 제외한 코드·설정·숫자·그림은 바이트를 보존했다.
- `historical/submission-20251215/`: 개인 과제 제출 tar의 평가 도구·최종 summary·그림. 중복 백업과 pyc는 제외했다. 출처가 다른 동명 도구를 임의로 섞지 않았다.
- raw pairwise CSV 8개는 의료 원문·참조답안·응답·reason 열을 제거하고 key·라벨·승자·model만 `decisions.csv`로 공개했다. 개인 연락처와 제3자 의료 문장은 재배포하지 않는다.
- 원 보고서는 개인정보가 있는 표지를 생략하고 metadata와 annotation을 제거한 별도 PDF로 공개했다. 원 인쇄 쪽 2~17과 당시 해석을 보존하며 [실험 정정](experiments.md)을 함께 제공한다.
- 손상된 2.49GB 환경 백업, 제3자 강의 PDF, 가중치·학습 원문, 비밀값, 개인 PC 경로는 공개하지 않는다. 원자료는 수정하지 않았다.
- `scripts/verify_historical_results.py`, `check_training_config.py`, 신규 tests/docs는 2026년 작업이다. 기존 실행 기록 검사기는 2026-09-09에 추가했다.

[provenance.json](provenance.json)은 복구 자료 122개 각각의 원본 SHA256, 공개 SHA256과 변환 내용을 기록한다. [change-manifest.json](change-manifest.json)은 이번 작업의 파일 변경 목록이다. 개인정보가 포함된 로컬 절대경로나 원본 파일명의 실명은 공개 manifest에 넣지 않았다.

데이터와 소스 전체에 임의의 MIT/Apache LICENSE를 부여하지 않았다. 모델 카드의 라이선스는 프로젝트나 제3자 데이터에 자동 적용되지 않는다. 데이터별 확인 범위는 [dataset](dataset.md)을 참조한다. 원 그래프의 보존은 그 안의 모든 과거 해석에 동의한다는 뜻이 아니다.
