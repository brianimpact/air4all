# AIR4*
여기 이미지 넣기(메인페이지)

[국문]

## Introduction
본 레포지토리는 재단법인 브라이언임팩트와 서울대학교가 진행한 **자기주도형 AI 학습 로드맵 개발 및 운영** 프로젝트의 일환으로 개발된 코드를 포함하고 있습니다.
해당 프로젝트의 결과물인 [aistudy.guide](http://aistudy.guide:3000)는 AI 관련 학습 주제들과 학습 자료들, 그리고 커리큘럼들을 제공하는 다언어(영문 및 국문 지원) 서비스입니다.
본 레포지토리의 각 하위 디렉토리에는 해당 서비스의 지속 가능성을 높일 수 있도록 구현된 자동화 모듈들이 실려 있습니다.

## Components
본 레포지토리는 *N*개의 하위 디렉토리로 구성되어 있으며, 각 디렉토리는 모듈에 대한 상세 설명(관련 연구, 요구 사항 등), 소스 코드, 그리고 입출력의 데이터 구조를 설명하고 있습니다.

각 모듈의 명칭과 역할은 아래와 같습니다:

### Study Unit Summarizer
Study Unit Summarizer는 학습 자료가 제공되는 AI 관련 세부 주제(학습 단위)에 대한 설명문을 생성합니다.
새로 서비스에 추가되는 주제에 대한 간략한 설명문을 작성하여 사용자가 해당 주제를 학습할 것인지 판단할 수 있게 보조하는 역할을 합니다.

### Content Classifier
Content Classifier는 학습 자료를 해당하는 학습 주제들과 학습 단위들로 분류합니다.
새로 서비스에 추가되는 학습 자료를 그에 해당하는 주제로 분류하여 서비스의 커버리지를 확장할 수 있게 보조합니다.

[영문]

## Introduction
This repository contains codes developed as part of the project called **Developing and Managing Self-Motivative AI Learning Roadmap**, conducted by Seoul National University and sponsored by Brian Impact Foundation.
The result of this project, [aistudy.guide](http://aistudy.guide:3000), is a multi-lingual service that provides structured AI-related study topics, materials, and curriculums.
Each subdirectory of this repository contains automating modules implemented to increase the sustainability of the service above.

## Components
This repository is composed of *N* subdirectories, where each subdirectory describes detailed description (including related works the module is based on, and requirements), source code, and I/O data structures for the module.

Brief introduction on the modules are as follows:

### Study Unit Summarizer
Study Unit Summarizer generates brief summary, or description, on a "study unit" (a low-level topic related to AI).
This module is designed to inform users on what each study unit will be about to aid them in selecting which topics to study.

### Content Classifier
Content Classifier labels each study material (content) with corresponding AI-related topics and study units.
