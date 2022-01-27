# AIR4*
여기 이미지 넣기(메인페이지)
[국문]
## Introduction
본 레포지토리는 재단법인 브라이언임팩트와 서울대학교가 진행한 [자기주도형 AI 학습 로드맵 개발 및 운영] 프로젝트의 일환으로 개발된 코드를 포함하고 있습니다.
해당 프로젝트의 결과물인 aistudy.guide는 AI 관련 학습 주제들과 학습 자료들, 그리고 커리큘럼들을 제시하는 서비스로, 해당 서비스의 지속 가능성을 위해 구현된 각 모듈의 코드는 각 하위 디렉토리에서 확인하실 수 있습니다.

## Components
본 레포지토리는 *개의 하위 디렉토리로 구성되어 있으며, 각 디렉토리는 모듈의 상세 설명(배경 연구, requirements 등), 구현 코드, 데이터를 포함하고 있습니다.
아래는 각 모듈의 명칭과 역할입니다.
### Study Unit Summarizer
Study Unit Summarizer는 학습 자료가 제공되는 AI 관련 세부 주제(학습 단위)에 대한 설명문을 생성합니다.
새로 서비스에 추가되는 주제에 대한 간략한 설명문을 작성하여 사용자가 해당 주제를 학습할 것인지 판단할 수 있게 보조하는 역할을 합니다.
### Content Classifier
Content Classifier는 학습 자료를 해당하는 학습 주제들과 학습 단위들로 분류합니다.
새로 서비스에 추가되는 학습 자료를 그에 해당하는 주제로 분류하여 서비스의 커버리지를 확장할 수 있게 보조합니다.

[영문]
## Introduction
This repository is a product of [Development of Self-Motivative Roadmap Service for AI Education], conducted by Seoul National University, sponsored by Brian Impact Foundation.
This project's result, aistudy.guide, is a multi-lingual service that provides structured AI-related study topics, materials, and curriculums, where the implementation of each module developed to make the service sustainable can be found under each subdirectory of this repository.

## Components
This repository is composed of * subdirectories, where each subdirectory consists of detailed description(research work the module is based on, requirements, etc.), code, and data for a module.
Brief introduction on the modules are as follows:
### Study Unit Summarizer
Study Unit Summarizer generates brief summary, or description, on a "study unit" (a low-level topic related to AI).
This module is designed to inform users on what each study unit will be about to aid them in selecting which topics to study.
### Content Classifier
Content Classifier labels each study material (content) with corresponding AI-related topics and study units.
