import software.amazon.awssdk.core.SdkBytes;
import software.amazon.awssdk.services.sagemakerruntime.SageMakerRuntimeClient;
import software.amazon.awssdk.services.sagemakerruntime.model.InvokeEndpointRequest;
import software.amazon.awssdk.services.sagemakerruntime.model.InvokeEndpointResponse;

public class Inference {
	public static void main(String[] args) {
		String requestBody = "{\"bucket\":\"nowfox\",\"image_uri\":\"data/zidane.jpg\",\"img_size\":416}";
		SdkBytes body = SdkBytes.fromUtf8String(requestBody);
		InvokeEndpointRequest request = InvokeEndpointRequest.builder().endpointName("yolov5")
				.contentType("application/json").body(body).build();
		SageMakerRuntimeClient client = SageMakerRuntimeClient.create();
		InvokeEndpointResponse response = client.invokeEndpoint(request);
		System.out.print(response.body().asUtf8String());
	}
}
