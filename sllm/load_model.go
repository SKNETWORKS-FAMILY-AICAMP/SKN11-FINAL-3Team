package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
)

func main() {
	fmt.Printf("시작")
	start := time.Now()

	bucket := "solar-10.7b"
	prefix := "base-model/solar/"
	localDir := "./solar-qlora-4bits"

	// AWS config 로드
	cfg, err := config.LoadDefaultConfig(context.TODO())
	if err != nil {
		log.Fatalf("AWS config 로드 실패: %v", err)
	}

	client := s3.NewFromConfig(cfg)

	// S3 객체 목록 가져오기
	paginator := s3.NewListObjectsV2Paginator(client, &s3.ListObjectsV2Input{
		Bucket: aws.String(bucket),
		Prefix: aws.String(prefix),
	})

	var keys []string
	for paginator.HasMorePages() {
		page, err := paginator.NextPage(context.TODO())
		if err != nil {
			log.Fatalf("객체 목록 가져오기 실패: %v", err)
		}
		for _, obj := range page.Contents {
			keys = append(keys, *obj.Key)
		}
	}

	fmt.Printf("📄 총 %d개 파일 다운로드 시작\n", len(keys))

	// 병렬 다운로드
	var wg sync.WaitGroup
	sem := make(chan struct{}, 10) // 병렬 다운로드 최대 10개

	for _, key := range keys {
		wg.Add(1)
		go func(key string) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			relPath := strings.TrimPrefix(key, prefix)
			localPath := filepath.Join(localDir, relPath)

			if err := os.MkdirAll(filepath.Dir(localPath), os.ModePerm); err != nil {
				log.Printf("디렉토리 생성 실패: %v", err)
				return
			}
			outFile, err := os.Create(localPath)
			if err != nil {
				log.Printf("파일 생성 실패: %v", err)
				return
			}
			defer outFile.Close()

			resp, err := client.GetObject(context.TODO(), &s3.GetObjectInput{
				Bucket: aws.String(bucket),
				Key:    aws.String(key),
			})
			if err != nil {
				log.Printf("다운로드 실패 (%s): %v", key, err)
				return
			}
			defer resp.Body.Close()

			n, err := io.Copy(outFile, resp.Body)
			if err != nil {
				log.Printf("파일 저장 실패 (%s): %v", key, err)
			} else {
				fmt.Printf("✅ %s 저장됨 (%d bytes)\n", localPath, n)
			}
		}(key)
	}

	wg.Wait()
	elapsed := time.Since(start)
	fmt.Printf("실행 시간: %.2f초\n", elapsed.Seconds())
}
