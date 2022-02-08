# AI Roadmap for All (AIR4*)
<a href="#">
  <img src="/img/banner.png" url= width="100%" height="auto">
  </a>

[🇰🇷 국문 소개](#소개) [🇱🇷 english introduction](#introduction)

## 소개
본 레포지토리는 재단법인 브라이언임팩트와 서울대학교가 진행한 **자기주도형 AI 학습 로드맵 개발 및 운영** 프로젝트의 일환으로 개발된 코드를 포함하고 있습니다.
해당 프로젝트의 결과물인 [aistudy.guide](http://aistudy.guide:3000)는 AI 관련 학습 주제들과 학습 자료들, 그리고 커리큘럼들을 제공하여 사용자들의 AI 학습을 돕는 다언어(영문 및 국문) 서비스입니다.
본 레포지토리의 각 하위 디렉토리에는 해당 서비스의 지속 가능성을 높일 수 있도록 구현된 자동화 모듈들이 실려 있습니다.

## 컴포넌트
본 레포지토리는 *N*개의 하위 디렉토리로 구성되어 있으며, 각 디렉토리는 모듈에 대한 상세 설명(관련 연구, 요구 사항 등), 소스 코드, 그리고 입출력의 데이터 구조를 설명하고 있습니다.

각 모듈의 명칭과 역할은 아래와 같습니다:

### 학습 단위 요약 모듈(Study Unit Summarizer)
Study Unit Summarizer는 학습 자료가 제공되는 AI 관련 세부 주제(학습 단위)에 대한 설명문을 생성합니다.
새로 서비스에 추가되는 주제에 대한 간략한 설명문을 작성하여 사용자가 해당 주제를 학습할 것인지 판단할 수 있게 보조하는 역할을 합니다.

### 학습 자료 분류 모듈(Content Classifier)
Content Classifier는 학습 자료를 해당하는 학습 주제들과 학습 단위들로 분류합니다.
새로 서비스에 추가되는 학습 자료를 그에 해당하는 주제로 분류하여 서비스의 커버리지를 확장할 수 있게 보조합니다.

### 학습 주제 그래프 확장 모듈(Topic Graph Expansion)
Topic Graph Expansion Module은 새로운 개념(학습 주제 또는 학습 단위)이 추가될 때 이를 기존 학습 주제 그래프에 통합하여 확장합니다. 
서비스의 커버리지를 확장하며 최신 학습 주제가 반영된 로드맵에서 학습할 수 있도록 보조합니다.

<br>

## Introduction
This repository contains codes developed as part of the project called **Developing and Managing Self-Motivative AI Learning Roadmap**, conducted by Seoul National University and sponsored by Brian Impact Foundation.
The result of this project, [aistudy.guide](http://aistudy.guide:3000), is a multi-lingual(english and korean) service that provides structured AI-related study topics, materials, and curriculums to help users learning AI.
Each subdirectory of this repository contains automating modules implemented to increase the sustainability of the service above.

## Components
This repository is composed of *N* subdirectories, where each subdirectory describes detailed description (including related works the module is based on, and requirements), source code, and I/O data structures for the module.

Brief introduction on the modules are as follows:

### Study Unit Summarizer
Study Unit Summarizer generates brief summary, or description, on a "study unit" (a low-level topic related to AI).
This module is designed to inform users on what each study unit will be about to aid them in selecting which topics to study.

### Content Classifier
Content Classifier labels each study material (content) with corresponding AI-related topics and study units.

### Topic Graph Expansion Module 
Topic Graph Expansion Module automatically expands the existing topic graph to incorporate a set of emerging concepts (topics or study units). 
It expands the coverage to help users learn from the roadmap that reflects the up-to-date topics.
