package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/PuerkitoBio/goquery"
	"github.com/ikawaha/kagome/tokenizer"
)

var (
	err error
)

const (
	inFile                = "../data/urls.txt"
	outFile               = "../data/output.txt"
	scrapingTargetElement = "div.article-body"
)

func main() {
	if err := run(); err != nil {
		fmt.Println("failed to run")
	}
	os.Exit(0)
}

func run() error {
	input, err := os.Open(inFile)
	if err != nil {
		fmt.Printf("failed to open %s", inFile)
		return err
	}
	defer input.Close()

	output, err := os.OpenFile(outFile, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		fmt.Printf("failed to open %s", outFile)
		return err
	}
	defer output.Close()

	r := csv.NewReader(input)
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			fmt.Printf("error caused by %s", err)
		}

		sbody, err := scraping(record[0])
		if err != nil {
			return err
		}

		formatedText := formatter(sbody)
		result := morphologicalAnalyzer(formatedText)

		fmt.Fprintln(output, result)
	}
	return nil
}

func scraping(url string) (targetElementText string, err error) {
	doc, err := goquery.NewDocument(url)
	if err != nil {
		fmt.Printf("failed to scraping. err=%s", err)
		return "", err
	}

	result := doc.Find(scrapingTargetElement).Text()

	return result, nil
}

func formatter(targetText string) (formatedText string) {
	sspace := strings.Replace(targetText, " ", "", -1)
	sLspace := strings.Replace(sspace, "　", "", -1)
	sReturn := strings.Replace(sLspace, "\n", "", -1)

	return sReturn
}

func morphologicalAnalyzer(formatedText string) (analyzedResult string) {
	result := ""
	dic := tokenizer.SysDicSimple()
	t := tokenizer.NewWithDic(dic)
	tokens := t.Tokenize(formatedText)
	for _, token := range tokens {
		sFeatures := token.Features()
		// debug出力用の文字列生成
		// features := strings.Join(sFeatures, ",")
		if cap(sFeatures) != 0 {
			// 名詞のみを抽出
			if sFeatures[0] == "名詞" {
				result += token.Surface + " "
			}
		}
	}
	return result
}
