package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

func main() {
	// 파일 열기
	inputPath := "../data/dialogue.json"
	outputPath := "../data/multiturn.json"

	data, err := ioutil.ReadFile(inputPath)
	if err != nil {
		log.Fatalf("입력 파일 읽기 실패: %v", err)
	}

	// json 파일 파싱
	var dialogues [][]Message
	if err := json.Unmarshal(data, &dialogues); err != nil {
		log.Fatalf("JSON 파싱 실패: %v", err)
	}

	// 멀티턴 변환
	var allSnapshots [][][]Message

	for _, dialogue := range dialogues {
		var context []Message
		var snapshots [][]Message

		for _, turn := range dialogue {
			context = append(context, turn)
			snapshot := make([]Message, len(context))
			copy(snapshot, context)
			snapshots = append(snapshots, snapshot)
		}

		allSnapshots = append(allSnapshots, snapshots)
	}

	// 저장
	result, err := json.MarshalIndent(allSnapshots, "", "  ")
	if err != nil {
		log.Fatalf("결과 JSON 변환 실패: %v", err)
	}

	if err := os.WriteFile(outputPath, result, 0644); err != nil {
		log.Fatalf("파일 저장 실패: %v", err)
	}

	fmt.Println("변환 완료 ✅ →", outputPath)
}
