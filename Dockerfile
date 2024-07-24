FROM ghcr.io/getzola/zola:v0.19.1 as zola

COPY . /myBlog
WORKDIR /myBlog
RUN ["zola", "build"]

FROM ghcr.io/static-web-server/static-web-server:2
WORKDIR /
COPY --from=zola /myBlog/public /public

# Expose the port that static-web-server will listen on
ENV PORT 8080
EXPOSE 8080

