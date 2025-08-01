import { useEffect, useState } from "react"
import { Result, Button, Card, Descriptions, Modal, List, Tag, Space, Typography } from "antd"
import { CheckCircleOutlined, FileTextOutlined, EyeOutlined, HomeOutlined } from "@ant-design/icons"

const { Title, Text } = Typography

export default function PendingPage({ user }) {
  const [botInfo, setBotInfo] = useState(null)
  const [showDataModal, setShowDataModal] = useState(false)

  useEffect(() => {
    const savedData = localStorage.getItem("lastBotRequest")
    if (savedData) {
      setBotInfo(JSON.parse(savedData))
    }
  }, [])

  const handleGoToDashboard = () => {
    window.location.href = "/"
  }

  const handleCreateAnother = () => {
    window.location.href = "/generate"
  }

  if (!botInfo) {
    return (
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          minHeight: "100vh",
        }}
      >
        <Card style={{ textAlign: "center", minWidth: "300px" }}>
          <Text>봇 생성 정보를 불러오는 중...</Text>
        </Card>
      </div>
    )
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        padding: "100px 20px",
      }}
    >
      <div style={{ maxWidth: "800px", margin: "0 auto", marginTop: "100px !important" }}>
        <Result
          icon={<CheckCircleOutlined style={{ color: "#52c41a", fontSize: "72px" }} />}
          title={
            <Title level={2} style={{ color: "#262626", marginBottom: "8px" }}>
              봇 생성 요청이 완료되었습니다!
            </Title>
          }
          subTitle={
            <div style={{ color: "#8c8c8c", fontSize: "16px", lineHeight: "1.6" }}>
              <Text>PDF를 분석 중입니다.</Text>
              <br />
              <Text>생성이 완료되면 이메일로 결과를 안내드릴게요.</Text>
            </div>
          }
          extra={
            <Space size="middle">
              <Button
                type="primary"
                size="large"
                icon={<HomeOutlined />}
                onClick={handleGoToDashboard}
                style={{
                  background: "#1890ff",
                  borderColor: "#1890ff",
                  height: "44px",
                  padding: "0 32px",
                  fontSize: "16px",
                }}
              >
                대시보드로 이동
              </Button>
              <Button
                size="large"
                onClick={handleCreateAnother}
                style={{
                  height: "44px",
                  padding: "0 32px",
                  fontSize: "16px",
                }}
              >
                새 봇 생성
              </Button>
            </Space>
          }
        />

        <Card
          title={
            <Space>
              <FileTextOutlined style={{ color: "#1890ff" }} />
              <span style={{ fontSize: "20px", fontWeight: "bold" }}>생성 요청 정보</span>
            </Space>
          }
          style={{
            marginTop: "40px",
            boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
            borderRadius: "12px",
          }}
          headStyle={{
            borderRadius: "12px 12px 0 0",
          }}
        >
          <Descriptions
            bordered
            column={1}
            labelStyle={{ fontWeight: "bold", width: "200px" }}
          >
            <Descriptions.Item label="회사명">{botInfo.company}</Descriptions.Item>
            <Descriptions.Item label="봇 이름">{botInfo.botName}</Descriptions.Item>
            <Descriptions.Item label="이메일">{botInfo.email}</Descriptions.Item>
            <Descriptions.Item label="고객센터 번호">{botInfo.cs_number}</Descriptions.Item>
            <Descriptions.Item label="인사말">{botInfo.first_text}</Descriptions.Item>
            <Descriptions.Item label="업로드한 파일 목록">
              <List
                dataSource={botInfo.files || []}
                renderItem={(file, index) => (
                  <List.Item key={index}>
                    <Text>{file.name}</Text>
                  </List.Item>
                )}
                locale={{ emptyText: "파일 없음" }}
              />
            </Descriptions.Item>
          </Descriptions>
        </Card>

        {/* 데이터 목록 모달 */}
        <Modal
          title={
            <Space>
              <FileTextOutlined style={{ color: "#1890ff" }} />
              <span>사용한 데이터 목록</span>
            </Space>
          }
          open={showDataModal}
          onCancel={() => setShowDataModal(false)}
          footer={[
            <Button key="close" onClick={() => setShowDataModal(false)}>
              닫기
            </Button>,
          ]}
          width={600}
          style={{ top: 20 }}
        >
          {Array.isArray(botInfo.files) && botInfo.files.length > 0 ? (
            <List
              dataSource={botInfo.files}
              renderItem={(file, index) => (
                <List.Item
                  style={{
                    padding: "12px 0",
                    borderBottom: index === botInfo.files.length - 1 ? "none" : "1px solid #f0f0f0",
                  }}
                >
                  <List.Item.Meta
                    avatar={<FileTextOutlined style={{ color: "#1890ff", fontSize: "16px" }} />}
                    title={<Text strong>{file.name}</Text>}
                    description={
                      <Space>
                        {file.size && <Text type="secondary">크기: {file.size}</Text>}
                        {file.type && <Tag color="geekblue">{file.type}</Tag>}
                      </Space>
                    }
                  />
                </List.Item>
              )}
            />
          ) : (
            <div
              style={{
                textAlign: "center",
                padding: "40px 0",
                color: "#8c8c8c",
              }}
            >
              <FileTextOutlined style={{ fontSize: "48px", marginBottom: "16px" }} />
              <br />
              <Text>저장된 파일 정보가 없습니다.</Text>
            </div>
          )}
        </Modal>
      </div>
    </div>
  )
}
