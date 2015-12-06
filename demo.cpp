#include <iostream>
#include <random>

#include <tclap/CmdLine.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "voronoi.h"

std::vector<float2> FillRandPoints(std::default_random_engine& gen, gpu::array<float2>& points, int x_max, int y_max) {
	std::uniform_real_distribution<> xdis(0.0, x_max);
	std::uniform_real_distribution<> ydis(0.0, y_max);

	std::vector<float2> cpu_points(points.size());

	for (int i=0; i < cpu_points.size(); ++i) {
		cpu_points[i].x = xdis(gen);
		cpu_points[i].y = ydis(gen);
	}

	std::cout << points.size() << std::endl;
	points.upload(cpu_points);

	return cpu_points;
}


void ProcessImage(const std::string& image_file) {
	std::random_device rand_dev;
	std::default_random_engine gen(rand_dev());

	cv::Mat image = cv::imread(image_file);
	image.convertTo(image, CV_8UC4);

	VoronoiBuffers buffer(image.rows, image.cols, 1600);

	std::vector<float2> points = FillRandPoints(gen, buffer.point_buffer, buffer.image_dims.x, buffer.image_dims.y);

	std::cout << "Image dimensions: " << buffer.image_dims.x << ", " << buffer.image_dims.y << std::endl;


	std::vector<uchar4> color_key;
	BuildIndex(image, points, color_key);
	std::cout << color_key.size() << ", " << buffer.color_key.size() << std::endl;
	buffer.color_key.upload(color_key);

	BruteForceVoronoi(buffer.id_image.raw_pointer(), buffer.image_dims, buffer.point_buffer.raw_pointer(), buffer.point_buffer.size());

	IndexToColors(buffer.id_image, buffer.color_image, buffer.color_key);


	cv::Mat output_image(buffer.image_dims.x, buffer.image_dims.y, CV_8UC4);
	buffer.color_image.download( (uchar4*) output_image.ptr());


	cv::imshow("Processed Image", output_image);
	cv::waitKey(-1);
}


int main(int argc, char* argv[]) {
	try {
		TCLAP::CmdLine cmd("Voronoi Demo", ' ', "0.0");
		TCLAP::UnlabeledValueArg<std::string> image("image", "image to use for demo", true, "", "image file");
		cmd.add(image);
		cmd.parse(argc, argv);

		ProcessImage(image.getValue());

	} catch (TCLAP::ArgException& e) {
		std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
	}

 	


	return 0;
}