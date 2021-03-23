import java.nio.ByteBuffer;

import com.amazonaws.services.sagemakerruntime.AmazonSageMakerRuntime;
import com.amazonaws.services.sagemakerruntime.AmazonSageMakerRuntimeClientBuilder;
import com.amazonaws.services.sagemakerruntime.model.InvokeEndpointRequest;
import com.amazonaws.services.sagemakerruntime.model.InvokeEndpointResult;

public class Inference {
	public static void main(String[] args) {
		String request = "{\"bucket\":\"nowfox\",\"image_uri\":\"data/zidane.jpg\",\"img_size\":416}";
		InvokeEndpointRequest invokeEndpointRequest = new InvokeEndpointRequest();
		invokeEndpointRequest.setContentType("application/json");
		ByteBuffer buf = ByteBuffer.wrap(request.getBytes());

		invokeEndpointRequest.setBody(buf);
		invokeEndpointRequest.setEndpointName("yolov5");
		invokeEndpointRequest.setAccept("application/json");

		AmazonSageMakerRuntime amazonSageMaker = AmazonSageMakerRuntimeClientBuilder.defaultClient();
		InvokeEndpointResult invokeEndpointResult = amazonSageMaker.invokeEndpoint(invokeEndpointRequest);
		byte[] response = invokeEndpointResult.getBody().array();
		String result = new String(response);
		System.out.print(result);
	}
}
